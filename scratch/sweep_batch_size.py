import subprocess
import time
import re
import sys
import os
import select

def main():
    batch_sizes = [1, 2, 4, 8, 16]
    results = {}

    base_dir = "/Users/lorenzoallegrini/Documents/MTP"
    training_script = os.path.join(base_dir, "src", "training.py")
    lora_path = os.path.join(base_dir, "saved_models", "mtp_backbone_lora_ff_w6")
    head_weights_path = os.path.join(base_dir, "saved_models", "mtp_head_ff_w6_final_.pth")

    for bs in batch_sizes:
        print(f"\n==========================================")
        print(f"Testing Batch Size: {bs}")
        print(f"==========================================")
        sys.stdout.flush()
        
        cmd = [
            ".venv/bin/python", training_script,
            "--head", "ff",
            "--window_size", "6",
            "--max_len", "2048",
            "--skip_phase_0", "true",
            "--skip_phase_1", "false",
            "--skip_phase_2", "false",
            "--max_samples", "15000",
            "--lora_path", lora_path,
            "--head_weights_path", head_weights_path,
            "--batch_size", str(bs),
            "--cheat"
        ]
        
        env = os.environ.copy()
        env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
            cwd=base_dir
        )
        
        start_time = time.time()
        last_speed = None
        oom_detected = False
        completed_steps = 0
        error_msg = ""
        timeout = 75  # ~1 minute run time plus startup overhead
        
        try:
            while True:
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    print("Timeout reached, terminating process...")
                    sys.stdout.flush()
                    break
                    
                ret = process.poll()
                if ret is not None:
                    if ret != 0:
                        # Process crashed, read the remaining stderr to diagnose
                        stderr_output = process.stderr.read()
                        print(f"Process crashed with exit code {ret}")
                        sys.stdout.flush()
                        if any(x in stderr_output.lower() for x in ["out of memory", "oom", "allocator", "allocation limit", "allocation failed"]):
                            oom_detected = True
                            print("OOM detected in stderr output!")
                            sys.stdout.flush()
                        else:
                            error_msg = stderr_output
                            print(f"Stderr error: {stderr_output[-500:]}")
                            sys.stdout.flush()
                    break
                    
                r, _, _ = select.select([process.stderr, process.stdout], [], [], 0.5)
                for src in r:
                    line = src.readline()
                    if not line:
                        continue
                    sys.stdout.write(f"[{bs}] {line}")
                    sys.stdout.flush()
                    
                    line_lower = line.lower()
                    if any(x in line_lower for x in ["out of memory", "oom", "allocation limit", "allocation failed"]):
                        oom_detected = True
                    if "runtimeerror" in line_lower or "error" in line_lower:
                        # Only keep relevant error messages, avoid cluttering
                        if not any(x in line_lower for x in ["userwarning", "futurewarning", "deprecationwarning"]):
                            error_msg += line
                        
                    # Parse tqdm line
                    match = re.search(r'Phase 1:.*\[.*,\s*([\d\.]+(?:s/it|it/s))', line)
                    if match:
                        last_speed = match.group(1)
                        completed_steps += 1
                        
                # We let it run for the whole minute (or up to timeout) to see the stable speed
                time.sleep(0.05)
                
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    
        if oom_detected:
            results[bs] = {"status": "OOM", "speed": None, "throughput": 0.0}
        elif error_msg and completed_steps == 0:
            # Clean up error message if it's too long
            clean_err = error_msg[-300:].strip() if len(error_msg) > 300 else error_msg.strip()
            results[bs] = {"status": f"ERROR: {clean_err}", "speed": None, "throughput": 0.0}
        else:
            throughput = 0.0
            if last_speed:
                val_str = re.sub(r'[^\d\.]', '', last_speed)
                try:
                    val = float(val_str)
                    if "s/it" in last_speed:
                        throughput = bs / val if val > 0 else 0.0
                    else:
                        throughput = bs * val
                except ValueError:
                    pass
            results[bs] = {"status": "SUCCESS" if completed_steps > 0 else "NO_STEPS", "speed": last_speed, "throughput": throughput}

    print("\n==========================================")
    print("SWEEP RESULTS SUMMARY")
    print("==========================================")
    for bs, res in results.items():
        print(f"Batch Size: {bs} | Status: {res['status']} | Speed: {res['speed']} | Throughput: {res['throughput']:.4f} samples/sec")
    sys.stdout.flush()

if __name__ == "__main__":
    main()

