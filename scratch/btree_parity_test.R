# BTree R<->Python parity check on the TRAINED weights.
# Proves the R circuit (forward + the get_full_vocab_dist draft path used by
# generate_draft / speculative decoding) is numerically equivalent to the Python
# forward, so R inference matches the semantics the model was trained with.
#
#   1. Python computes reference per-position marginals on a fixed hidden state.
#   2. R loads the same hidden state + weights and compares.
#
# Run from the project root:  Rscript scratch/btree_parity_test.R

PYBIN = ".venv/bin/python"

# (1) Python reference -------------------------------------------------------
py = '
import sys; sys.modules["torchvision"]=None; sys.path.append("src")
import torch, numpy as np
from models.probabilistic_circuits import BTree
from models.mtp_llm import MTP_LLM
m=MTP_LLM("google/byt5-small", BTree, window_size=6, ranks=32,
          lora_path="saved_models/lora_btree_w6/mtp_backbone_lora_btree_w6", cheat=False)
m.heads.load_state_dict(torch.load("saved_models/mtp_head_btree_w6_final.pth", map_location="cpu"))
m.eval()
d=int(m.embed_dim)
idx=torch.arange(d, dtype=torch.float32)
h=((torch.sin(idx*0.013)+torch.cos(idx*0.07))*0.5).view(1,1,d)
np.save("/tmp/btree_h.npy", h.numpy())
with torch.no_grad():
    ref=m._circuit(h).exp()[0,0].numpy()   # [W,V] marginals
np.save("/tmp/btree_ref.npy", ref)
print("python ref ok", ref.shape)
'
Sys.setenv(TOKENIZERS_PARALLELISM = "false", PYTHONWARNINGS = "ignore")
status = system2(PYBIN, args = c("-c", shQuote(py)))
if (status != 0) stop("python reference generation failed")

# (2) R comparison -----------------------------------------------------------
suppressMessages({source("mtpc/llm.R"); source("mtpc/utils.R"); source("utils.R")
                  source("mtpc/probabilistic_circuits.R")})
np = import("numpy"); torch = import("torch")
h   = torch$tensor(np$load("/tmp/btree_h.npy"))$to(torch$float32)
ref = np$load("/tmp/btree_ref.npy")                       # [6,384]

m = LLMWrapper(model_id = "google/byt5-small", head_type = "btree", window_size = 6L, ranks = 32L,
               lora_path = "saved_models/lora_btree_w6/mtp_backbone_lora_btree_w6", cheat = FALSE)
m$load_weights("saved_models/mtp_head_btree_w6_final.pth", device = "cpu", shift_offset_minus_1 = FALSE)
m$eval()
h = h$to(m$backbone$device)

# (a) R forward vs Python forward
logm  = with(torch$no_grad(), m$circuit$forward(m, h))
Rfwd  = as.array(logm$exp()$cpu()$numpy())[1, 1, , ]      # [6,384]
d1    = max(abs(Rfwd - ref))

# (b) R draft path (get_draft_probs + get_full_vocab_dist) vs Python forward
probs  = m$circuit$get_draft_probs(m, h)
Rdraft = t(sapply(1:6, function(s) m$circuit$get_full_vocab_dist(probs, s, 1L)))
d2     = max(abs(Rdraft - ref))

cat(sprintf("\n[PARITY] max|R.forward - Py.forward|         = %.3e\n", d1))
cat(sprintf("[PARITY] max|R.get_full_vocab_dist - Py|    = %.3e\n", d2))
cat(if (d1 < 1e-4 && d2 < 1e-4)
      "\n==== PARITY PASS: R BTree == Python BTree on trained weights ====\n"
    else "\n!!!! PARITY MISMATCH !!!!\n")
