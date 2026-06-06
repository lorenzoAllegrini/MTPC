import sys
sys.modules['torchvision'] = None

import torch
import torch.nn as nn
import os
import gc
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from contextlib import nullcontext

from models.probabilistic_circuits import FF, CanonicPolyidiac, MTPC_HMM, BTree
from models.mtp_llm import MTP_LLM
from utils import MTPChatDataset, compute_mtpc_loss, get_byt5_preprocess_function, load_tulu_dataset, CHAT_TEMPLATE, get_model_paths_python

def swap_model_head(model, head_class, window_size, ranks, device):
    """
    Dynamically swaps the probabilistic head in the MTP_LLM container.
    """
    circuit = head_class(
        embedding_size=model.embed_dim, 
        vocabulary_size=model.vocab_size, 
        window_size=window_size,
        **({'ranks': ranks} if 'ranks' in head_class.__init__.__code__.co_varnames else {})
    )
    
    if head_class.__name__ in ('MTPC_HMM', 'BTree'):
        layers = {
            "sum_unit_omega_init": circuit.sum_unit_omega_init,
            "input_units_phi": circuit.input_units_phi,
            "sum_unit_omega_transitions": circuit.sum_unit_omega_transitions
        }
    elif head_class.__name__ == 'FF':
        layers = {f"input_units_phi_{i+1}": layer for i, layer in enumerate(circuit.input_units_phi)}
    elif head_class.__name__ == 'CanonicPolyidiac':
        layers = {
            "sum_unit_omega": circuit.sum_unit_omega,
            "input_units_phi": circuit.input_units_phi
        }
    else:
        layers = {}
        
    model.heads = nn.ModuleDict(layers)
    model._circuit = circuit
    model._head_class_name = head_class.__name__
    model.to(device)

def init_emissions_from_stp(model, window_size, ranks):
    """
    Initializes emission weights from T5 backbone's output embeddings (STP lm_head).
    """
    with torch.no_grad():
        if hasattr(model.backbone, 'lm_head'):
            stp_weight = model.backbone.lm_head.weight.detach().clone()
        else:
            stp_weight = model.backbone.get_output_embeddings().weight.detach().clone()
            
        V, H = stp_weight.shape
        head_name = model._head_class_name

        if head_name in ('MTPC_HMM', 'BTree'):
            # BTree shares the HMM emission layout [window, ranks, vocab]
            emission_init = (stp_weight
                .unsqueeze(0).unsqueeze(0)
                .expand(window_size, ranks, -1, -1)
                .reshape(-1, H))
            model.heads['input_units_phi'].weight.copy_(emission_init)
            nn.init.zeros_(model.heads['input_units_phi'].bias)

        elif head_name == 'FF':
            for i in range(1, window_size + 1):
                layer = model.heads[f'input_units_phi_{i}']
                layer.weight.copy_(stp_weight)
                nn.init.zeros_(layer.bias)

        elif head_name == 'CanonicPolyidiac':
            emission_init = (stp_weight
                .unsqueeze(0).unsqueeze(0)
                .expand(ranks, window_size, -1, -1)
                .reshape(-1, H))
            model.heads['input_units_phi'].weight.copy_(emission_init)
            nn.init.zeros_(model.heads['input_units_phi'].bias)

def init_cp_from_ff(model, ff_state_dict, window_size, ranks, device):
    """
    Initializes CP head's emission weights from a trained Feed-Forward (FF) head.
    This ensures mathematical continuity between Phase 1 and Phase 2.
    """
    with torch.no_grad():
        H = model.embed_dim
        V = model.vocab_size
        W = window_size
        R = ranks
        
        # CP input_units_phi.weight shape: [R * W * V, H]
        new_emissions = torch.zeros(W, R, V, H, device=device, dtype=model.backbone.dtype)
        
        for t in range(W):
            ff_key = f'input_units_phi_{t+1}.weight'
            old_ff_key = f'emission_{t+1}.weight'
            if ff_key in ff_state_dict:
                ff_weight = ff_state_dict[ff_key].to(device=device, dtype=model.backbone.dtype) # [V, H]
                new_emissions[t, :, :, :] = ff_weight.unsqueeze(0).expand(R, V, H)
            elif old_ff_key in ff_state_dict:
                ff_weight = ff_state_dict[old_ff_key].to(device=device, dtype=model.backbone.dtype) # [V, H]
                new_emissions[t, :, :, :] = ff_weight.unsqueeze(0).expand(R, V, H)
            else:
                print(f"[WARNING] Key {ff_key} or {old_ff_key} not found in FF state_dict.")
                
        # Load weights into the input_units_phi layer of CP
        reshaped_emissions = new_emissions.permute(1, 0, 2, 3).reshape(-1, H)
        
        # Add tiny symmetry-breaking noise
        noise = torch.randn_like(reshaped_emissions) * 1e-4
        reshaped_emissions += noise
        
        model.heads['input_units_phi'].weight.copy_(reshaped_emissions)
        nn.init.zeros_(model.heads['input_units_phi'].bias)

        # Initialize sum_unit_omega (gate) to zero to ensure uniform weight distribution initially
        nn.init.zeros_(model.heads['sum_unit_omega'].weight)
        nn.init.zeros_(model.heads['sum_unit_omega'].bias)

def init_hmm_from_ff(model, ff_state_dict, window_size, ranks, device):
    """
    Initializes the HMM head's emission weights from a trained Feed-Forward (FF) head, so the
    HMM starts equivalent to FF (slides: "...initialise the emission matrices of the target
    circuit (CP / HMM)"). The transitions keep their identity init from the circuit constructor
    and the initial-state gate is set to uniform, so the HMM initially reproduces the FF marginals.
    """
    with torch.no_grad():
        H = model.embed_dim
        V = model.vocab_size
        W = window_size
        R = ranks
        # HMM input_units_phi.view layout is [window, ranks, vocab] (no permute, unlike CP).
        new_emissions = torch.zeros(W, R, V, H, device=device, dtype=model.backbone.dtype)
        for t in range(W):
            ff_key = f'input_units_phi_{t+1}.weight'
            old_ff_key = f'emission_{t+1}.weight'
            if ff_key in ff_state_dict:
                ff_weight = ff_state_dict[ff_key].to(device=device, dtype=model.backbone.dtype)  # [V, H]
            elif old_ff_key in ff_state_dict:
                ff_weight = ff_state_dict[old_ff_key].to(device=device, dtype=model.backbone.dtype)
            else:
                print(f"[WARNING] FF key for window position {t+1} not found in FF state_dict.")
                continue
            new_emissions[t, :, :, :] = ff_weight.unsqueeze(0).expand(R, V, H)

        reshaped_emissions = new_emissions.reshape(-1, H)
        reshaped_emissions += torch.randn_like(reshaped_emissions) * 1e-4  # symmetry-breaking
        model.heads['input_units_phi'].weight.copy_(reshaped_emissions)
        nn.init.zeros_(model.heads['input_units_phi'].bias)

        # Uniform initial-state distribution (transitions stay identity-initialised from the circuit).
        nn.init.zeros_(model.heads['sum_unit_omega_init'].weight)
        nn.init.zeros_(model.heads['sum_unit_omega_init'].bias)

def init_btree_from_ff(model, ff_state_dict, window_size, ranks, device):
    """
    Initializes the BTree head from a trained Feed-Forward (FF) head so the tree reduces to FF at
    init (the user's plan): copy each window-position's FF emission into every rank (HMM layout
    [window, ranks, vocab]) with tiny symmetry-breaking noise, and set ALL hierarchical sum gates
    (root prior + tree transitions) to zero -> a perfectly uniform mixture at every node of the tree.
    """
    with torch.no_grad():
        H = model.embed_dim
        V = model.vocab_size
        W = window_size
        R = ranks
        new_emissions = torch.zeros(W, R, V, H, device=device, dtype=model.backbone.dtype)
        for t in range(W):
            ff_key = f'input_units_phi_{t+1}.weight'
            old_ff_key = f'emission_{t+1}.weight'
            if ff_key in ff_state_dict:
                ff_weight = ff_state_dict[ff_key].to(device=device, dtype=model.backbone.dtype)  # [V, H]
            elif old_ff_key in ff_state_dict:
                ff_weight = ff_state_dict[old_ff_key].to(device=device, dtype=model.backbone.dtype)
            else:
                print(f"[WARNING] FF key for window position {t+1} not found in FF state_dict.")
                continue
            new_emissions[t, :, :, :] = ff_weight.unsqueeze(0).expand(R, V, H)

        reshaped_emissions = new_emissions.reshape(-1, H)
        reshaped_emissions += torch.randn_like(reshaped_emissions) * 1e-4  # break rank symmetry
        model.heads['input_units_phi'].weight.copy_(reshaped_emissions)
        nn.init.zeros_(model.heads['input_units_phi'].bias)

        # Uniform mixture at every sum node (root prior + all tree transitions -> 0 logits).
        nn.init.zeros_(model.heads['sum_unit_omega_init'].weight)
        nn.init.zeros_(model.heads['sum_unit_omega_init'].bias)
        nn.init.zeros_(model.heads['sum_unit_omega_transitions'].weight)
        nn.init.zeros_(model.heads['sum_unit_omega_transitions'].bias)

def plot_losses(losses, phase_name, save_path):
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(losses, label="Loss")
        plt.title(f"{phase_name} Loss Curve")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        print(f"[SYSTEM] {phase_name} loss plot saved to {save_path}")
    except Exception as e:
        print(f"[WARNING] Failed to generate {phase_name} loss plot: {e}")

def load_head_weights(model_heads, weights_path, device, shift_offset_minus_1=False, target_window_size=None, ranks=None):
    state_dict = torch.load(weights_path, map_location=device)
    new_state_dict = {}
    
    # 1. Determine model type from model_heads (ModuleDict) keys
    is_cp = "sum_unit_omega" in model_heads
    is_hmm = "sum_unit_omega_init" in model_heads
    is_ff = any(k.startswith("input_units_phi_") for k in model_heads.keys())
    
    # 2. Determine target_window_size and ranks
    if target_window_size is None:
        if is_ff:
            target_indices = []
            for k in model_heads.keys():
                if k.startswith("input_units_phi_"):
                    try:
                        target_indices.append(int(k.split("_")[-1].split(".")[0]))
                    except:
                        pass
            target_window_size = max(target_indices) if target_indices else 4
        else:
            target_window_size = 4  # Default fallback
            
    if ranks is None:
        if is_cp:
            ranks = model_heads["sum_unit_omega"].out_features
        elif is_hmm:
            ranks = model_heads["sum_unit_omega_init"].out_features
        else:
            ranks = 32  # Default fallback
            
    # 3. Map keys in checkpoint to canonical keys
    canonical_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        if k.startswith("gate."):
            new_k = k.replace("gate.", "sum_unit_omega.", 1)
        elif k.startswith("emission_projs."):
            new_k = k.replace("emission_projs.", "input_units_phi.", 1)
        elif k.startswith("init_gate."):
            new_k = k.replace("init_gate.", "sum_unit_omega_init.", 1)
        elif k.startswith("emissions."):
            new_k = k.replace("emissions.", "input_units_phi.", 1)
        elif k.startswith("transitions."):
            new_k = k.replace("transitions.", "sum_unit_omega_transitions.", 1)
            
        # FF model checkpoint conversion (emission_ -> input_units_phi_)
        if k.startswith("emission_") and (".weight" in k or ".bias" in k):
            parts = k.split(".", 1)
            src_num = int(parts[0].split("_")[1])
            new_k = f"input_units_phi_{src_num}.{parts[1]}"
            
        canonical_state_dict[new_k] = v
        
    # 4. Perform shift offset loading if requested
    for k, v in canonical_state_dict.items():
        if not shift_offset_minus_1:
            new_state_dict[k] = v
            continue
            
        # FF model shifting
        if is_ff and k.startswith("input_units_phi_") and (".weight" in k or ".bias" in k):
            parts = k.split(".", 1)
            src_num = int(parts[0].split("_")[-1])
            suffix = parts[1]
            new_num = src_num - 1
            if new_num >= 1 and new_num <= target_window_size:
                new_k = f"input_units_phi_{new_num}.{suffix}"
                new_state_dict[new_k] = v
            # Fallback for the last head
            if new_num == target_window_size - 1:
                new_k = f"input_units_phi_{target_window_size}.{suffix}"
                new_state_dict[new_k] = v.clone()
                
        # CP model shifting
        elif is_cp and k.startswith("input_units_phi."):
            vocab_size = model_heads["input_units_phi"].out_features // (ranks * target_window_size)
            is_bias = (".bias" in k)
            
            if is_bias:
                src_window_size = v.shape[0] // (ranks * vocab_size)
                v_reshaped = v.view(ranks, src_window_size, vocab_size)
                new_tensor = torch.zeros(ranks, target_window_size, vocab_size, device=v.device, dtype=v.dtype)
            else:
                src_window_size = v.shape[0] // (ranks * vocab_size)
                v_reshaped = v.view(ranks, src_window_size, vocab_size, v.shape[1])
                new_tensor = torch.zeros(ranks, target_window_size, vocab_size, v.shape[1], device=v.device, dtype=v.dtype)
                
            for new_idx in range(target_window_size):
                src_idx = new_idx + 1
                if src_idx < src_window_size:
                    new_tensor[:, new_idx] = v_reshaped[:, src_idx]
                else:
                    new_tensor[:, new_idx] = v_reshaped[:, src_window_size - 1]
                    
            new_state_dict[k] = new_tensor.view(-1) if is_bias else new_tensor.view(-1, v.shape[1])
            
        # HMM model shifting
        elif is_hmm and k.startswith("input_units_phi."):
            vocab_size = model_heads["input_units_phi"].out_features // (ranks * target_window_size)
            is_bias = (".bias" in k)
            
            if is_bias:
                src_window_size = v.shape[0] // (ranks * vocab_size)
                v_reshaped = v.view(src_window_size, ranks, vocab_size)
                new_tensor = torch.zeros(target_window_size, ranks, vocab_size, device=v.device, dtype=v.dtype)
            else:
                src_window_size = v.shape[0] // (ranks * vocab_size)
                v_reshaped = v.view(src_window_size, ranks, vocab_size, v.shape[1])
                new_tensor = torch.zeros(target_window_size, ranks, vocab_size, v.shape[1], device=v.device, dtype=v.dtype)
                
            for new_idx in range(target_window_size):
                src_idx = new_idx + 1
                if src_idx < src_window_size:
                    new_tensor[new_idx] = v_reshaped[src_idx]
                else:
                    new_tensor[new_idx] = v_reshaped[src_window_size - 1]
                    
            new_state_dict[k] = new_tensor.view(-1) if is_bias else new_tensor.view(-1, v.shape[1])
            
        # HMM transitions shifting
        elif is_hmm and k.startswith("sum_unit_omega_transitions."):
            is_bias = (".bias" in k)
            
            if is_bias:
                src_transitions = v.shape[0] // (ranks * ranks)
                v_reshaped = v.view(src_transitions, ranks, ranks)
                new_tensor = torch.zeros(target_window_size - 1, ranks, ranks, device=v.device, dtype=v.dtype)
            else:
                src_transitions = v.shape[0] // (ranks * ranks)
                v_reshaped = v.view(src_transitions, ranks, ranks, v.shape[1])
                new_tensor = torch.zeros(target_window_size - 1, ranks, ranks, v.shape[1], device=v.device, dtype=v.dtype)
                
            for new_idx in range(target_window_size - 1):
                src_idx = new_idx + 1
                if src_idx < src_transitions:
                    new_tensor[new_idx] = v_reshaped[src_idx]
                else:
                    new_tensor[new_idx] = v_reshaped[src_transitions - 1]
                    
            new_state_dict[k] = new_tensor.view(-1) if is_bias else new_tensor.view(-1, v.shape[1])
            
        else:
            new_state_dict[k] = v
            
    missing, unexpected = model_heads.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"[WARNING] Missing keys when loading weights: {missing}")
    if unexpected:
        print(f"[WARNING] Unexpected keys when loading weights: {unexpected}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="MTP Speculative Joint Training Pipeline")
    parser.add_argument("--model_id", type=str, default="google/byt5-small", help="Pretrained model ID")
    parser.add_argument("--max_samples", type=int, default=10, help="Max dataset samples")
    parser.add_argument("--max_len", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--window_size", type=int, default=4, help="Speculative window size")
    parser.add_argument("--ranks", type=int, default=32, help="Ranks for probabilistic circuit")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--head", type=str, choices=["cp", "hmm", "ff", "btree"], default="cp", help="Target head class (cp = CanonicPolyidiac, hmm = MTPC_HMM, ff = FF, btree = BTree)")
    parser.add_argument("--skip_phase_0", type=str, default="true", choices=["true", "false", "auto"], help="Skip Phase 0 autoregressive backbone fine-tuning. 'auto' skips if checkpoint exists.")
    parser.add_argument("--skip_phase_1", type=str, default="false", choices=["true", "false", "auto"], help="Skip Phase 1 FF warm-up. 'auto' skips if checkpoint exists.")
    parser.add_argument("--skip_phase_2", type=str, default="false", choices=["true", "false", "auto"], help="Skip Phase 2 joint training. 'auto' skips if checkpoint exists.")
    parser.add_argument("--use_pretrain", type=str, default="true", choices=["true", "false"], help="Load pretrained Phase 0 LoRA weights if skipping Phase 0. Default is true.")
    parser.add_argument("--shift_offset", action="store_true", help="Shift pre-trained head weights by -1 to correct alignment sfasamento.")
    parser.add_argument("--lora_path", type=str, default=None, help="Custom path to LoRA adapter checkpoint directory to load.")
    parser.add_argument("--head_weights_path", type=str, default=None, help="Custom path to speculative head weights .pth file to load.")
    parser.add_argument("--cheat", action="store_true", help="Enable cheating mode (feed generated tokens to encoder during SFT and validation).")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "mps", "cpu"], help="Compute device (default: cuda).")
    parser.add_argument("--amp", action="store_true", help="Enable bf16 mixed-precision autocast (recommended on A100/Ampere+ GPUs).")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers (raise to keep the GPU fed; 0 = main thread).")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory for checkpoints + tokenized cache (default: <repo>/saved_models). Set to a Google Drive path to persist models.")
    parser.add_argument("--head_lr", type=float, default=3e-4, help="Phase-2 learning rate for the speculative head (raise for a quick loss-drop sanity test).")
    parser.add_argument("--warmup_steps", type=int, default=400, help="Phase-2 linear warmup steps for the backbone LoRA (lower for short runs).")
    args = parser.parse_args()
    args.use_pretrain = (args.use_pretrain == "true")

    model_id = args.model_id
    # Use the requested device, falling back gracefully if it is unavailable.
    if args.device == "cuda" and not torch.cuda.is_available():
        fallback = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"[WARNING] --device cuda requested but CUDA is unavailable; using '{fallback}'.")
        args.device = fallback
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print("[WARNING] --device mps requested but MPS is unavailable; using 'cpu'.")
        args.device = "cpu"
    device = torch.device(args.device)

    # Performance: enable TF32 matmuls on Ampere+ (negligible precision impact, free speedup).
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    # bf16 autocast context (params stay fp32 -> no head/backbone dtype clash; softmax/logsumexp/
    # loss stay fp32 -> stable). No GradScaler needed for bf16.
    amp_ctx = (torch.autocast(device_type=device.type, dtype=torch.bfloat16)
               if (args.amp and device.type in ("cuda", "cpu")) else nullcontext())
    print(f"Device: {device} | Model: {model_id} | amp={'bf16' if args.amp else 'off'}")
    
    # 1. DATASET SETUP
    max_samples = args.max_samples
    max_len = args.max_len
    window_size = args.window_size
    ranks = args.ranks
    
    if args.head == "cp":
        target_head_class = CanonicPolyidiac
    elif args.head == "hmm":
        target_head_class = MTPC_HMM
    elif args.head == "ff":
        target_head_class = FF
    elif args.head == "btree":
        target_head_class = BTree

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Where to read/write checkpoints (and the tokenized cache). Default: <repo>/saved_models.
    # Pass --save_dir to point this at a Google Drive folder (e.g. when the repo is cloned to
    # the fast local /content disk but you want models persisted on Drive).
    save_dir = args.save_dir if args.save_dir else os.path.join(base_dir, "saved_models")
    os.makedirs(save_dir, exist_ok=True)
    print(f"[SYSTEM] Checkpoints will be saved to: {save_dir}")

    # Cache tokenized dataset on disk to avoid re-tokenizing on every trial (includes max_len in key)
    tokenized_path = os.path.join(save_dir, f"tokenized_train_data_{max_samples}_len{max_len}")
    if os.path.exists(tokenized_path):
        from datasets import load_from_disk
        print(f"[SYSTEM] Loading tokenized dataset from disk cache: {tokenized_path}...")
        train_data = load_from_disk(tokenized_path)
    else:
        print(f"\n[SYSTEM] Loading raw dataset with max_samples={max_samples}...")
        train_data, val_data = load_tulu_dataset("ai2-adapt-dev/flan_v2_converted", max_samples=max_samples)
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.chat_template = CHAT_TEMPLATE
        preprocess_fn = get_byt5_preprocess_function(tokenizer, max_len, tokenizer.chat_template)
        
        train_data = train_data.map(
            preprocess_fn,
            batched=True,
            num_proc=4,
            remove_columns=train_data.column_names,
            desc="Tokenizing train data"
        )
        print(f"[SYSTEM] Saving tokenized dataset to disk cache: {tokenized_path}...")
        train_data.save_to_disk(tokenized_path)
    
    batch_size = args.batch_size
    accumulation_steps = max(1, 16 // batch_size)
    print(f"[SYSTEM] Using batch_size={batch_size} with gradient_accumulation_steps={accumulation_steps} (Effective batch_size = {batch_size * accumulation_steps})")

    train_chat_dataset = MTPChatDataset(train_data)
    train_loader = DataLoader(
        train_chat_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,                 # feed the GPU in parallel (0 starves it)
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
    )
    
    # Toggle to skip Phase 2 (Joint training), Phase 0 (Backbone SFT) and/or Phase 1 (FF Warm-up) and load existing weights
    head_name_str = args.head  # cp | hmm | ff | btree  (no longer collapse non-cp/hmm to "ff")
    phase2_lora_path, phase2_head_path = get_model_paths_python(head_name_str, window_size, save_dir)

    if args.skip_phase_2 == "auto":
        checkpoint_exists = os.path.exists(phase2_lora_path) and os.path.exists(phase2_head_path)
        skip_phase_2 = checkpoint_exists
        if checkpoint_exists:
            print(f"[SYSTEM] Auto-detected Phase 2 checkpoints. Setting skip_phase_2=True.")
        else:
            print(f"[SYSTEM] No Phase 2 checkpoints found. Setting skip_phase_2=False.")
    else:
        skip_phase_2 = (args.skip_phase_2 == "true")

    resume_lora_path = None
    if skip_phase_2:
        if os.path.exists(phase2_lora_path):
            resume_lora_path = phase2_lora_path
            print(f"[SYSTEM] Found Phase 2 backbone adapter at {resume_lora_path}")
            skip_phase_0 = True
            skip_phase_1 = True
        else:
            print(f"[SYSTEM] Phase 2 adapter not found at {phase2_lora_path}. Cannot skip Phase 2.")
            skip_phase_2 = False

    if not skip_phase_2:
        if args.skip_phase_0 == "auto":
            checkpoint_exists = os.path.exists(os.path.join(save_dir, "byt5_standard_lora_phase0"))
            skip_phase_0 = checkpoint_exists
            if checkpoint_exists:
                print(f"[SYSTEM] Auto-detected Phase 0 checkpoint at {os.path.join(save_dir, 'byt5_standard_lora_phase0')}. Setting skip_phase_0=True.")
            else:
                print(f"[SYSTEM] No Phase 0 checkpoint found. Setting skip_phase_0=False.")
        else:
            skip_phase_0 = (args.skip_phase_0 == "true")
            
        if args.skip_phase_1 == "auto":
            phase1_paths = [
                os.path.join(save_dir, f"lora_{head_name_str}_w{window_size}", f"mtp_backbone_lora_{head_name_str}_w{window_size}"),
                os.path.join(save_dir, f"lora_{head_name_str}_w{window_size}_phase1"),
                os.path.join(save_dir, f"lora_ff_w{window_size}_phase1"),  # Phase 1 is always FF; CP/HMM reuse its backbone
            ]
            final_lora_path, _ = get_model_paths_python(head_name_str, window_size, save_dir)
            if final_lora_path and final_lora_path not in phase1_paths:
                phase1_paths.append(final_lora_path)
            if args.shift_offset:
                final_lora_path_w_plus_1, _ = get_model_paths_python(head_name_str, window_size + 1, save_dir)
                if final_lora_path_w_plus_1 and final_lora_path_w_plus_1 not in phase1_paths:
                    phase1_paths.append(final_lora_path_w_plus_1)
                    
            checkpoint_exists = any(os.path.exists(path) for path in phase1_paths)
            skip_phase_1 = checkpoint_exists
            if checkpoint_exists:
                print(f"[SYSTEM] Auto-detected Phase 1 checkpoint. Setting skip_phase_1=True.")
            else:
                print(f"[SYSTEM] No Phase 1 checkpoint found. Setting skip_phase_1=False.")
        else:
            skip_phase_1 = (args.skip_phase_1 == "true")
        
        if skip_phase_1:
            phase1_paths = [
                os.path.join(save_dir, f"lora_{head_name_str}_w{window_size}", f"mtp_backbone_lora_{head_name_str}_w{window_size}"),
                os.path.join(save_dir, f"lora_{head_name_str}_w{window_size}_phase1"),
                os.path.join(save_dir, f"lora_ff_w{window_size}_phase1"),  # Phase 1 is always FF; CP/HMM reuse its backbone
            ]
            final_lora_path, _ = get_model_paths_python(head_name_str, window_size, save_dir)
            if final_lora_path and final_lora_path not in phase1_paths:
                phase1_paths.append(final_lora_path)
            if args.shift_offset:
                final_lora_path_w_plus_1, _ = get_model_paths_python(head_name_str, window_size + 1, save_dir)
                if final_lora_path_w_plus_1 and final_lora_path_w_plus_1 not in phase1_paths:
                    phase1_paths.append(final_lora_path_w_plus_1)
                    
            found_path = False
            for path in phase1_paths:
                if os.path.exists(path):
                    resume_lora_path = path
                    found_path = True
                    print(f"[SYSTEM] Found Phase 1 backbone adapter at {resume_lora_path}")
                    break
            if not found_path:
                print(f"[SYSTEM] Phase 1 adapter not found in any of {phase1_paths}. Falling back to standard LoRA Phase 0 checkpoint.")
                phase0_paths = [
                    os.path.join(save_dir, "byt5_standard_lora_phase0"),
                    os.path.join(save_dir, "byt5_standard_lora_verifier", "byt5_standard_lora")
                ]
                for path in phase0_paths:
                    if os.path.exists(path):
                        resume_lora_path = path
                        break
                skip_phase_1 = False  # Need to run Phase 1 if we only have Phase 0 checkpoint
        elif skip_phase_0:
            # Prioritize matching speculative backbone LoRA adapters, falling back to standard SFT backbone
            spec_lora_path, _ = get_model_paths_python(head_name_str, window_size, save_dir)
            
            # Fallback to window_size + 1 if shift_offset is active
            if (not spec_lora_path or not os.path.exists(spec_lora_path)) and args.shift_offset:
                alt_spec_lora_path, _ = get_model_paths_python(head_name_str, window_size + 1, save_dir)
                if alt_spec_lora_path and os.path.exists(alt_spec_lora_path):
                    spec_lora_path = alt_spec_lora_path
            
            phase0_paths = []
            if spec_lora_path and os.path.exists(spec_lora_path):
                phase0_paths.append(spec_lora_path)
            
            alt_spec_lora_path = os.path.join(save_dir, f"lora_{head_name_str}_w{window_size}_phase1")
            if os.path.exists(alt_spec_lora_path):
                phase0_paths.append(alt_spec_lora_path)
                
            if args.shift_offset:
                alt_spec_lora_path_w6 = os.path.join(save_dir, f"lora_{head_name_str}_w{window_size+1}_phase1")
                if os.path.exists(alt_spec_lora_path_w6):
                    phase0_paths.append(alt_spec_lora_path_w6)
                
            phase0_paths.extend([
                os.path.join(save_dir, "byt5_standard_lora_verifier", "byt5_standard_lora"),
                os.path.join(save_dir, "byt5_standard_lora_phase0")
            ])
            
            found_path = False
            for path in phase0_paths:
                if os.path.exists(path):
                    resume_lora_path = path
                    found_path = True
                    break
            if not found_path:
                resume_lora_path = None
            
    if not args.use_pretrain:
        resume_lora_path = None
        print("[SYSTEM] Using default/fresh backbone without loading pretrained LoRA weights (pass --use_pretrain to load).")

    if args.lora_path:
        resume_lora_path = args.lora_path
        print(f"[SYSTEM] Overriding LoRA path with custom value: {resume_lora_path}")

    print(f"[SYSTEM] skip_phase_0={skip_phase_0}, skip_phase_1={skip_phase_1}, skip_phase_2={skip_phase_2}, resume_lora_path={resume_lora_path}")
    
    # Instantiate the base speculative model container
    model = MTP_LLM(
        model_id=model_id, 
        head_class=target_head_class, 
        window_size=window_size,
        ranks=ranks,
        lora_path=resume_lora_path if (skip_phase_0 or skip_phase_1 or skip_phase_2) else None,
        cheat=args.cheat
    )
    model.to(device)
    
    if hasattr(model.backbone, "gradient_checkpointing_enable"):
        print("[SYSTEM] Enabling gradient checkpointing on the backbone to save memory...")
        model.backbone.gradient_checkpointing_enable()
        if hasattr(model.backbone, "enable_input_require_grads"):
            model.backbone.enable_input_require_grads()

    # ==========================================================================
    # PHASE 0: AUTOREGRESSIVE LLM BACKBONE FINE-TUNING ONLY
    # ==========================================================================
    if skip_phase_0:
        print("\n[SYSTEM] Skipping PHASE 0: Pre-trained LoRA backbone loaded successfully.")
    else:
        print("\n--- [PHASE 0] Standard LoRA Backbone Autoregressive Fine-Tuning ---")
        model.backbone.train()
        model.heads.eval()
        
        lora_params = [p for p in model.backbone.parameters() if p.requires_grad]
        optimizer = Adam(lora_params, lr=5e-4)
        
        pbar = tqdm(train_loader, desc="Phase 0")
        phase0_losses = []
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Mask assistant tokens in the encoder input to prevent future token leakage during SFT unless cheating is active
            if args.cheat:
                encoder_input_ids = input_ids
            else:
                decoder_start_token_id = model.backbone.config.decoder_start_token_id
                encoder_input_ids = input_ids.clone()
                encoder_input_ids[labels != -100] = decoder_start_token_id
            
            # Standard Seq2Seq forward pass through the backbone (calculates autoregressive loss)
            with amp_ctx:
                outputs = model.backbone(
                    input_ids=encoder_input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss

            # Normalize the loss by gradient accumulation steps
            loss = loss / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
                if device.type == "mps":
                    torch.mps.empty_cache()
                
            pbar.set_postfix(loss=f"{(loss.item() * accumulation_steps):.4f}")
            phase0_losses.append(loss.item() * accumulation_steps)

        print("[PHASE 0] Completed successfully.")
        
        # Checkpoint automatico per la Fase 0
        phase0_save_path = os.path.join(save_dir, "byt5_standard_lora_phase0")
        model.backbone.save_pretrained(phase0_save_path)
        print(f"\n[SYSTEM] Phase 0 Backbone LoRA adapter salvato in: {phase0_save_path}\n")
        
        # Plot phase 0 loss
        plot_losses(phase0_losses, "Phase 0", os.path.join(save_dir, "loss_phase0.png"))

    # ==========================================================================
    # PHASE 1: FEED-FORWARD (FF) WARM-UP (Backbone LoRA Active)
    # ==========================================================================
    if skip_phase_1:
        print("\n[SYSTEM] Skipping PHASE 1: Pre-tuned FF backbone LoRA loaded successfully.")
    else:
        print("\n--- [PHASE 1] Speculative Warm-up with Feed-Forward (FF) Head (Backbone Active) ---")
        # Swap head to Feed-Forward and initialize emissions
        swap_model_head(model, FF, window_size, ranks, device)
        
        # Resume from a previously-trained FF head if one exists, so re-running on more data
        # CONTINUES instead of restarting. Priority: explicit path > final > phase-1 checkpoint.
        # (To also keep the backbone progress, run with --skip_phase_0 true so the matching FF
        # LoRA backbone is reloaded; otherwise the head lands on a freshly re-trained backbone.)
        loaded_ff_head = False
        _, ff_final_path = get_model_paths_python("ff", window_size, save_dir)
        ff_candidates = []
        if args.head_weights_path:
            ff_candidates.append(args.head_weights_path)
        ff_candidates.append(ff_final_path)
        ff_candidates.append(os.path.join(save_dir, f"mtp_head_ff_w{window_size}_phase1.pth"))
        if args.shift_offset:
            _, alt_path = get_model_paths_python("ff", window_size + 1, save_dir)
            ff_candidates.append(alt_path)

        for cand in ff_candidates:
            if cand and os.path.exists(cand):
                print(f"[SYSTEM] Resuming Phase 1 from existing FF head: {cand}")
                try:
                    load_head_weights(model.heads, cand, device, shift_offset_minus_1=args.shift_offset, target_window_size=window_size, ranks=ranks)
                    loaded_ff_head = True
                    break
                except Exception as e:
                    print(f"[WARNING] Failed to load FF head weights from {cand}: {e}")

        if not loaded_ff_head:
            print("[SYSTEM] No existing FF head found; initializing emissions from backbone STP.")
            init_emissions_from_stp(model, window_size, ranks)
        
        model.backbone.train()
        model.heads.train()
        
        ff_params = list(model.heads.parameters())
        lora_params = [p for p in model.backbone.parameters() if p.requires_grad]
        param_groups = [
            {"params": ff_params, "lr": 1e-3},
            {"params": lora_params, "lr": 1e-4}
        ]
        optimizer = Adam(param_groups)
        
        pbar = tqdm(train_loader, desc="Phase 1")
        phase1_losses = []
        warmup_steps = 200
        base_lora_lr = 1e-4
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Linear warmup for backbone LoRA parameter group
            if batch_idx < warmup_steps:
                current_lora_lr = base_lora_lr * (batch_idx / warmup_steps)
                optimizer.param_groups[1]['lr'] = current_lora_lr
            else:
                optimizer.param_groups[1]['lr'] = base_lora_lr
                
            with amp_ctx:
                _, mtp_logits, _ = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = compute_mtpc_loss(mtp_logits, labels, window_size, gamma=0.8, is_log_probs=False)

            # Normalize the loss by gradient accumulation steps
            loss = loss / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
                if device.type == "mps":
                    torch.mps.empty_cache()
                
            pbar.set_postfix(loss=f"{(loss.item() * accumulation_steps):.4f}", lr_lora=f"{optimizer.param_groups[1]['lr']:.2e}")
            phase1_losses.append(loss.item() * accumulation_steps)

        print("[PHASE 1] Completed successfully.")
        
        # Checkpoint automatico per la Fase 1
        ff_head_save_path = os.path.join(save_dir, f"mtp_head_ff_w{window_size}_phase1.pth")
        torch.save(model.heads.state_dict(), ff_head_save_path)
        ff_lora_save_path = os.path.join(save_dir, f"lora_ff_w{window_size}_phase1")
        model.backbone.save_pretrained(ff_lora_save_path)
        print(f"\n[SYSTEM] Phase 1 FF Head salvata in: {ff_head_save_path}")
        print(f"[SYSTEM] Phase 1 Backbone LoRA adapter salvato in: {ff_lora_save_path}\n")
        
        # Plot phase 1 loss
        plot_losses(phase1_losses, "Phase 1", os.path.join(save_dir, "loss_phase1.png"))

    # ==========================================================================
    # PHASE 2: TARGET PROBABILISTIC HEAD JOINT TRAINING
    # ==========================================================================
    if skip_phase_2:
        print(f"\n[SYSTEM] Skipping PHASE 2: Pre-trained target head and LoRA backbone loaded successfully.")
        # Swap head to target probabilistic circuit
        swap_model_head(model, target_head_class, window_size, ranks, device)
        # Load final head weights
        head_load_path = args.head_weights_path if args.head_weights_path else phase2_head_path
        if not args.head_weights_path and not os.path.exists(head_load_path) and args.shift_offset:
            _, alt_head_path = get_model_paths_python(head_name_str, window_size + 1, save_dir)
            if os.path.exists(alt_head_path):
                head_load_path = alt_head_path
                
        if os.path.exists(head_load_path):
            print(f"[SYSTEM] Loading final head weights from {head_load_path}...")
            try:
                load_head_weights(model.heads, head_load_path, device, shift_offset_minus_1=args.shift_offset, target_window_size=window_size, ranks=ranks)
            except Exception as e:
                print(f"[WARNING] Failed to load final head weights: {e}")
        else:
            print(f"[WARNING] Phase 2 head weights not found at {head_load_path}!")
    else:
        print(f"\n--- [PHASE 2] Target {target_head_class.__name__} Joint Training with Differential LRs ---")
        # Clear memory from previous phase
        if 'optimizer' in locals():
            del optimizer
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
            
        # Swap head to target probabilistic circuit and initialize emissions
        swap_model_head(model, target_head_class, window_size, ranks, device)
        
        # Try to load pre-trained target head weights first if available
        head_loaded = False
        _, target_head_path = get_model_paths_python(head_name_str, window_size, save_dir)
        
        if skip_phase_1:
            # If Phase 1 was skipped, args.head_weights_path represents the starting head weights
            if args.head_weights_path:
                target_head_path = args.head_weights_path
            elif not os.path.exists(target_head_path) and args.shift_offset:
                _, alt_head_path = get_model_paths_python(head_name_str, window_size + 1, save_dir)
                if os.path.exists(alt_head_path):
                    target_head_path = alt_head_path
        else:
            # Phase 1 was executed. If we are joint training the FF head, load the Phase 1 save.
            if target_head_class.__name__ == 'FF':
                target_head_path = os.path.join(save_dir, f"mtp_head_ff_w{window_size}_phase1.pth")
            else:
                # If target head is CP/HMM, we do not load the Phase 1 FF head directly as target head.
                # Only check the standard target head path if it exists.
                if not os.path.exists(target_head_path):
                    target_head_path = None
                    
        # Check if the target weights file matches the target head type
        is_valid_target_checkpoint = False
        if target_head_path and os.path.exists(target_head_path):
            try:
                chk = torch.load(target_head_path, map_location="cpu")
                has_ff_keys = any(k.startswith("input_units_phi_") or k.startswith("emission_") for k in chk.keys())
                has_cp_keys = "input_units_phi.weight" in chk or "gate.weight" in chk or "sum_unit_omega.weight" in chk
                has_hmm_keys = "sum_unit_omega_init.weight" in chk or "init_gate.weight" in chk or "sum_unit_omega_transitions.weight" in chk
                
                if target_head_class.__name__ == 'CanonicPolyidiac':
                    is_valid_target_checkpoint = has_cp_keys and not has_ff_keys
                elif target_head_class.__name__ == 'MTPC_HMM':
                    is_valid_target_checkpoint = has_hmm_keys and not has_ff_keys
                elif target_head_class.__name__ == 'FF':
                    is_valid_target_checkpoint = has_ff_keys
                else:
                    is_valid_target_checkpoint = True
            except Exception as e:
                print(f"[WARNING] Could not pre-verify checkpoint keys: {e}")
                is_valid_target_checkpoint = False
                
        if is_valid_target_checkpoint and target_head_path and os.path.exists(target_head_path):
            print(f"[SYSTEM] Loading pre-trained target head weights from {target_head_path} for joint training...")
            try:
                load_head_weights(model.heads, target_head_path, device, shift_offset_minus_1=args.shift_offset, target_window_size=window_size, ranks=ranks)
                head_loaded = True
            except Exception as e:
                print(f"[WARNING] Failed to load pre-trained target head weights: {e}")
                
        # If no pre-trained final weights, initialise the target circuit from the trained Phase 1
        # FF emissions (CP and HMM both start equivalent to FF, per the paper/slides).
        if not head_loaded and target_head_class.__name__ in ('CanonicPolyidiac', 'MTPC_HMM', 'BTree'):
            if not skip_phase_1:
                ff_head_save_path = os.path.join(save_dir, f"mtp_head_ff_w{window_size}_phase1.pth")
            else:
                ff_head_save_path = args.head_weights_path if args.head_weights_path else os.path.join(save_dir, f"mtp_head_ff_w{window_size}_phase1.pth")

            if os.path.exists(ff_head_save_path):
                print(f"[SYSTEM] Transferring trained Phase 1 FF weights from {ff_head_save_path} to {target_head_class.__name__}...")
                try:
                    ff_state_dict = torch.load(ff_head_save_path, map_location=device)
                    if target_head_class.__name__ == 'CanonicPolyidiac':
                        init_cp_from_ff(model, ff_state_dict, window_size, ranks, device)
                    elif target_head_class.__name__ == 'BTree':
                        init_btree_from_ff(model, ff_state_dict, window_size, ranks, device)
                    else:
                        init_hmm_from_ff(model, ff_state_dict, window_size, ranks, device)
                    head_loaded = True
                except Exception as e:
                    print(f"[WARNING] Failed to transfer FF head weights: {e}")
                    
        if not head_loaded:
            print(f"[SYSTEM] Initializing {target_head_class.__name__} weights from backbone STP...")
            init_emissions_from_stp(model, window_size, ranks)
        
        model.backbone.train()
        model.heads.train()
        
        is_log_probs = target_head_class.__name__ in ['CanonicPolyidiac', 'MTPC_HMM', 'BTree']
        
        # Differential learning rates for joint fine-tuning (head_lr / warmup_steps overridable via CLI)
        param_groups = [
            {"params": list(model.heads.parameters()), "lr": args.head_lr},
            {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": 2e-5}
        ]
        optimizer = Adam(param_groups)

        pbar = tqdm(train_loader, desc="Phase 2")
        phase2_losses = []
        warmup_steps = args.warmup_steps
        start_lora_lr = 1e-6
        target_lora_lr = 1e-4
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Linear warmup for backbone LoRA parameter group (from 1e-5 to 1e-4)
            if batch_idx < warmup_steps:
                current_lora_lr = start_lora_lr + (target_lora_lr - start_lora_lr) * (batch_idx / warmup_steps)
                optimizer.param_groups[1]['lr'] = current_lora_lr
            else:
                optimizer.param_groups[1]['lr'] = target_lora_lr
                
            with amp_ctx:
                _, mtp_logits, _ = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = compute_mtpc_loss(mtp_logits, labels, window_size, gamma=0.8, is_log_probs=is_log_probs)

            # Normalize the loss by gradient accumulation steps
            loss = loss / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
                if device.type == "mps":
                    torch.mps.empty_cache()
                
            pbar.set_postfix(loss=f"{(loss.item() * accumulation_steps):.4f}", lr_lora=f"{optimizer.param_groups[1]['lr']:.2e}")
            phase2_losses.append(loss.item() * accumulation_steps)

        print("[PHASE 2] Completed successfully.")
        
        # Plot phase 2 loss
        plot_losses(phase2_losses, "Phase 2", os.path.join(save_dir, "loss_phase2.png"))
    
    # ==========================================================================
    # SAVING THE FINAL WEIGHTS & ADAPTERS
    # ==========================================================================
    head_type = model._head_class_name
    head_name_str = "cp" if head_type == "CanonicPolyidiac" else ("hmm" if head_type == "MTPC_HMM" else head_type.lower())
    filename = f"mtp_head_{head_name_str}_w{window_size}_final.pth"

    os.makedirs(save_dir, exist_ok=True)  # reuse save_dir resolved above (respects --save_dir)
    save_path = os.path.join(save_dir, filename)
    torch.save(model.heads.state_dict(), save_path)
    print(f"\n[SYSTEM] Speculative head ({head_type}) weights saved to {save_path}")
    
    lora_save_path = os.path.join(save_dir, f"lora_{head_name_str}_w{window_size}", f"mtp_backbone_lora_{head_name_str}_w{window_size}")
    model.backbone.save_pretrained(lora_save_path)
    print(f"[SYSTEM] Backbone LoRA adapters saved to {lora_save_path}")
    
    # ==========================================================================
    # QUICK QUALITATIVE TEST ON A FEW SAMPLES
    # ==========================================================================
    print("\n" + "="*80)
    print(" [TESTING CONFIGURATION SUMMARY]")
    print(f"  - Target Head Class:  {target_head_class.__name__}")
    print(f"  - Backbone LoRA Path:  {resume_lora_path}")
    if skip_phase_2:
        print(f"  - Head Weights Path:  {head_load_path if 'head_load_path' in locals() else 'Not defined'}")
    else:
        if 'head_loaded' in locals() and head_loaded:
            print(f"  - Head Weights Path:  {target_head_path if 'target_head_path' in locals() else 'Not defined'}")
        else:
            print("  - Head Weights Path:  Initialized from backbone STP / FF Phase 1 Transfer")
    print("="*80 + "\n")
    
    print("\n" + "="*60)
    print("[SYSTEM] Running qualitative validation test on a couple of samples...")
    print("="*60)
    
    
    model.eval()
    test_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    with torch.no_grad():
        count = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass to get logits and hidden states (passing labels to prevent leakage and align decoder)
            _, mtp_logits, hidden_states = model(input_ids, attention_mask=attention_mask, labels=labels)
            
            for b in range(input_ids.shape[0]):
                valid_indices = (labels[b] != -100).nonzero(as_tuple=True)[0]
                if len(valid_indices) == 0:
                    continue
                
                # Pick a test index in the middle of the assistant response
                test_idx = valid_indices[len(valid_indices) // 2].item()
                
                # Context up to test_idx
                context_ids = input_ids[b, :test_idx]
                # True next tokens (ground truth for window): step 1 predicts labels[test_idx]
                true_ids = labels[b, test_idx : test_idx + window_size]
                
                # Get logits/log_probs at test_idx
                is_log_probs = head_type in ['CanonicPolyidiac', 'MTPC_HMM', 'BTree']
                sample_logits = mtp_logits[b, test_idx] # [W, V]
                
                if is_log_probs:
                    sample_log_probs = sample_logits
                else:
                    import torch.nn.functional as F
                    sample_log_probs = F.log_softmax(sample_logits, dim=-1)
                
                # Predict speculative window of tokens
                if head_type in ['CanonicPolyidiac', 'MTPC_HMM', 'BTree']:
                    # Use THIS example's hidden state (b), not the whole batch then [0] — otherwise
                    # the sampled draft comes from batch item 0 while the printed marginal is item b,
                    # making the sampled tokens look like ~0-probability garbage.
                    test_emb = hidden_states[b:b+1, test_idx:test_idx+1, :]
                    sampled_ids = model._circuit.generate_draft(test_emb)[0]
                else:
                    sampled_ids = sample_logits.argmax(dim=-1)
                
                argmax_ids = sample_logits.argmax(dim=-1)
                
                # Filter ignore tokens
                true_ids_clean = true_ids[true_ids != -100]
                sampled_ids_clean = sampled_ids[sampled_ids != -100]
                argmax_ids_clean = argmax_ids[argmax_ids != -100]
                
                context_text = test_tokenizer.decode(context_ids)
                true_text = test_tokenizer.decode(true_ids_clean)
                sampled_text = test_tokenizer.decode(sampled_ids_clean)
                argmax_text = test_tokenizer.decode(argmax_ids_clean)
                
                print(f"\n--- SAMPLE {count + 1} ---")
                print(f"CONTEXT (last 80 bytes): ...{context_text[-80:]!r}")
                print(f"GROUND TRUTH (window {window_size}): {true_text!r} (Bytes: {true_ids_clean.tolist()})")
                print(f"PREDICTED (ARGMAX)  (window {window_size}): {argmax_text!r} (Bytes: {argmax_ids_clean.tolist()})")
                print(f"PREDICTED (SAMPLED) (window {window_size}): {sampled_text!r} (Bytes: {sampled_ids_clean.tolist()})")
                
                # Step-by-step probabilities and loss
                print("Step-by-step Probabilities & Loss:")
                gamma = 0.8
                discounted_loss_sum = 0.0
                actual_window = min(window_size, len(true_ids))
                
                for j in range(actual_window):
                    t_id = true_ids[j].item()
                    s_id = sampled_ids[j].item()
                    a_id = argmax_ids[j].item()
                    
                    if t_id == -100:
                        continue
                        
                    log_p_true = sample_log_probs[j, t_id].item()
                    log_p_samp = sample_log_probs[j, s_id].item()
                    log_p_amax = sample_log_probs[j, a_id].item()
                    
                    p_true = torch.exp(torch.tensor(log_p_true)).item()
                    p_samp = torch.exp(torch.tensor(log_p_samp)).item()
                    p_amax = torch.exp(torch.tensor(log_p_amax)).item()
                    
                    step_loss = -log_p_true
                    discounted_loss_sum += (gamma ** j) * step_loss
                    
                    t_char = test_tokenizer.decode([t_id])
                    s_char = test_tokenizer.decode([s_id])
                    a_char = test_tokenizer.decode([a_id])
                    
                    print(f"  Step {j+1}: True={t_char!r} (p={p_true:.6f}, loss={step_loss:.4f}) | Argmax={a_char!r} (p={p_amax:.6f}) | Sampled={s_char!r} (p={p_samp:.6f})")
                
                print(f"Sample MTP Loss (discounted sum): {discounted_loss_sum:.4f}")
                
                count += 1
                if count >= 50:
                    break
            if count >= 50:
                break
                
    print("[SYSTEM] Training pipeline finished successfully.")

if __name__ == "__main__":
    main()
