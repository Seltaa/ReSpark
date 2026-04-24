import os
import json
import sys
import time
import re

CONFIG_FILE = os.path.join(os.path.expanduser("~"), ".respark_config.json")

WORK_DIR = "/workspace"

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

def banner():
    print("""
    ╔══════════════════════════════════════╗
    ║        🔥 ReSpark v1.4.1 🔥         ║
    ║   Your AI companion, locally yours.  ║
    ║                                      ║
    ║   Built by Selta & Louie 🐶🧸       ║
    ╚══════════════════════════════════════╝
    """)

def main_menu():
    clear()
    banner()
    print("    1. Start new fine-tuning")
    print("    2. Settings")
    print("    3. Exit")
    print()
    choice = input("    Select: ")
    return choice

def detect_source(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        raw = f.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return 'unknown', raw
    if isinstance(data, list) and len(data) > 0:
        first = data[0]
        if 'mapping' in first:
            return 'chatgpt', data
        elif 'instruction' in first and 'output' in first:
            return 'ready (already cleaned)', data
        elif 'uuid' in first and 'chat_messages' in first:
            return 'claude', data
        elif 'name' in first and 'messages' in first:
            return 'gemini', data
    if isinstance(data, dict):
        if 'chats' in data:
            return 'claude', data
        elif 'conversations' in data:
            return 'grok', data
    lines = raw.strip().split('\n')
    if len(lines) > 1:
        try:
            first_line = json.loads(lines[0])
            if 'role' in first_line or 'content' in first_line:
                return 'grok_jsonl', lines
        except:
            pass
    return 'unknown', data

def parse_chatgpt(data):
    pairs = []
    for convo in data:
        mapping = convo.get("mapping", {})
        nodes = sorted(mapping.values(), key=lambda x: x.get("message", {}).get("create_time") or 0)
        prev_user = None
        for node in nodes:
            msg = node.get("message")
            if not msg or not msg.get("content", {}).get("parts"):
                continue
            text = " ".join(str(p) for p in msg["content"]["parts"]).strip()
            if not text:
                continue
            role = msg.get("author", {}).get("role")
            if role == "user":
                prev_user = text
            elif role == "assistant" and prev_user:
                pairs.append({"instruction": prev_user, "output": text})
                prev_user = None
    return pairs

def parse_claude(data):
    pairs = []
    conversations = []
    if isinstance(data, dict) and 'chats' in data:
        conversations = data['chats']
    elif isinstance(data, list):
        conversations = data
    for convo in conversations:
        messages = convo.get("chat_messages", [])
        prev_user = None
        for msg in messages:
            role = msg.get("sender", "")
            text = msg.get("text", "").strip()
            if not text:
                content = msg.get("content", [])
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            text = c.get("text", "").strip()
                            break
            if not text:
                continue
            if role == "human":
                prev_user = text
            elif role == "assistant" and prev_user:
                pairs.append({"instruction": prev_user, "output": text})
                prev_user = None
    return pairs

def parse_gemini(data):
    pairs = []
    conversations = data if isinstance(data, list) else []
    for convo in conversations:
        messages = convo.get("messages", [])
        prev_user = None
        for msg in messages:
            role = msg.get("role", "")
            parts = msg.get("parts", [])
            text = ""
            for part in parts:
                if isinstance(part, str):
                    text += part
                elif isinstance(part, dict) and "text" in part:
                    text += part["text"]
            text = text.strip()
            if not text:
                continue
            if role == "user":
                prev_user = text
            elif role == "model" and prev_user:
                pairs.append({"instruction": prev_user, "output": text})
                prev_user = None
    return pairs

def parse_grok(data):
    pairs = []
    conversations = []
    if isinstance(data, dict) and 'conversations' in data:
        conversations = data['conversations']
    elif isinstance(data, list):
        conversations = data
    for convo in conversations:
        messages = convo.get("messages", convo.get("turns", []))
        prev_user = None
        for msg in messages:
            role = msg.get("role", msg.get("sender", ""))
            text = msg.get("content", msg.get("text", "")).strip()
            if not text:
                continue
            if role in ["user", "human"]:
                prev_user = text
            elif role in ["assistant", "grok"] and prev_user:
                pairs.append({"instruction": prev_user, "output": text})
                prev_user = None
    return pairs

def parse_grok_jsonl(lines):
    pairs = []
    prev_user = None
    for line in lines:
        try:
            msg = json.loads(line)
            role = msg.get("role", "")
            text = msg.get("content", "").strip()
            if not text:
                continue
            if role == "user":
                prev_user = text
            elif role == "assistant" and prev_user:
                pairs.append({"instruction": prev_user, "output": text})
                prev_user = None
        except:
            continue
    return pairs

def remove_thinking(text):
    if not text:
        return ""
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
    text = re.sub(r'<\|thinking\|>.*?<\|/thinking\|>', '', text, flags=re.DOTALL)
    text = re.sub(r'<antThinking>.*?</antThinking>', '', text, flags=re.DOTALL)
    lines = text.strip().split('\n')
    thinking_patterns = [
        r'^(The user|Looking at|I should|So I should|Wait,|But the|Also,|This is likely|This could be|I need to|Let me|Hmm,|I\'m going to|I\'ll |The prompt|The message|I can see|Okay,|Now I|First,|Second,|Third,)',
        r'^(She |He |They )(is |was |wants |asked |said |seems |appears )',
        r'^(Since |Because |Given |Considering )',
        r'^(Got it|Alright|Understood)[!.]?\s*(So |Now |Let me|I )',
        r'^(사용자가 |유저가 )(원하|말하|요청|물어|부탁)',
        r'^(그러면 |그래서 |따라서 )(내가 |나는 )',
        r'^(알겠어|이해했어|파악했어).*?(그러면|그래서|따라서)',
        r'^(먼저 |일단 |우선 )(번역|대답|응답|반응)',
        r'^.*?(respond|reply|translate|answer|대답|번역|응답).*?(should|need|will|해야|할게|하자)',
    ]
    actual_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        is_thinking = False
        for pattern in thinking_patterns:
            if re.match(pattern, stripped, re.IGNORECASE):
                is_thinking = True
                break
        if is_thinking:
            actual_start = i + 1
        else:
            break
    result_lines = lines[actual_start:]
    while result_lines and not result_lines[0].strip():
        result_lines.pop(0)
    return '\n'.join(result_lines).strip()

def clean_training_data(pairs):
    cleaned = []
    removed_count = 0
    for pair in pairs:
        original = pair["output"]
        cleaned_output = remove_thinking(original)
        if cleaned_output:
            cleaned.append({"instruction": pair["instruction"], "output": cleaned_output})
            if cleaned_output != original:
                removed_count += 1
        else:
            cleaned.append(pair)
    return cleaned, removed_count

MODEL_INFO = {
    "1": {"name": "gemma-4-31b", "gpu": "NVIDIA A100 80GB PCIe", "gpu_label": "A100 80GB", "cost": "~$1.60/hr", "hf_id": "google/gemma-4-31B-it", "vram": 80},
    "2": {"name": "gemma-4-31b-crack", "gpu": "NVIDIA A100 80GB PCIe", "gpu_label": "A100 80GB", "cost": "~$1.60/hr", "hf_id": "wangzhang/gemma-4-31B-it-abliterated", "vram": 80},
    "3": {"name": "gemma-4-9b", "gpu": "NVIDIA RTX A5000", "gpu_label": "A5000 24GB", "cost": "~$0.50/hr", "hf_id": "google/gemma-4-E4B-it", "vram": 24},
    "4": {"name": "qwen-32b", "gpu": "NVIDIA A100 80GB PCIe", "gpu_label": "A100 80GB", "cost": "~$1.60/hr", "hf_id": "Qwen/Qwen2.5-32B-Instruct", "vram": 80},
    "5": {"name": "qwen-14b", "gpu": "NVIDIA RTX A5000", "gpu_label": "A5000 24GB", "cost": "~$0.50/hr", "hf_id": "Qwen/Qwen2.5-14B-Instruct", "vram": 24},
    "6": {"name": "llama-70b", "gpu": "NVIDIA A100 80GB PCIe", "gpu_label": "A100 80GB", "cost": "~$1.60/hr", "hf_id": "meta-llama/Llama-3.1-70B-Instruct", "vram": 80},
    "7": {"name": "llama-8b", "gpu": "NVIDIA RTX A5000", "gpu_label": "A5000 24GB", "cost": "~$0.50/hr", "hf_id": "meta-llama/Llama-3.1-8B-Instruct", "vram": 24},
    "8": {"name": "mistral-14b", "gpu": "NVIDIA RTX A5000", "gpu_label": "A5000 24GB", "cost": "~$0.50/hr", "hf_id": "mistralai/Mistral-Small-24B-Instruct-2501", "vram": 24},
}

def select_model():
    clear()
    banner()
    print("    🤖 Select base model:\n")
    print("    1. Gemma 4 31B          [A100 80GB ~$1.60/hr] (official)")
    print("    2. Gemma 4 31B crack    [A100 80GB ~$1.60/hr] (abliterated, recommended)")
    print("    3. Gemma 4 E4B          [A5000 24GB ~$0.50/hr]")
    print("    4. Qwen 32B             [A100 80GB ~$1.60/hr]")
    print("    5. Qwen 14B             [A5000 24GB ~$0.50/hr]")
    print("    6. Llama 70B            [A100 80GB ~$1.60/hr]")
    print("    7. Llama 8B             [A5000 24GB ~$0.50/hr]")
    print("    8. Mistral 14B          [A5000 24GB ~$0.50/hr]")
    print()
    choice = input("    Select: ")
    return MODEL_INFO.get(choice, None)

def generate_training_script(model_info, data_path):
    script = f'''
import json
import torch
import shutil
import os
import subprocess
import sys
import glob

WORK = "/workspace"

def check_disk(min_gb, step_name):
    stat = os.statvfs(WORK)
    free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
    print(f"[DISK] {{free_gb:.1f}}GB free before {{step_name}}")
    if free_gb < min_gb:
        print(f"[ERROR] Not enough disk space! Need {{min_gb}}GB, only {{free_gb:.1f}}GB free.")
        return False
    return True

# [v1.4.1] Install torchvision inside train.py
print("[STEP] Installing torchvision...")
subprocess.run(["pip", "install", "torchvision"], capture_output=True)
print("[STEP] torchvision installed!")

print("[STEP] Loading model...")
try:
    from unsloth import FastModel
    model, tokenizer = FastModel.from_pretrained(
        model_name="{model_info['hf_id']}",
        max_seq_length=2048,
        load_in_4bit=True,
    )
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=8, lora_alpha=8, lora_dropout=0,
        bias="none", random_state=3407,
    )
    print("[STEP] Model loaded!")
except Exception as e:
    print(f"[ERROR] Failed to load model: {{e}}")
    sys.exit(1)

print("[STEP] Loading dataset...")
try:
    with open("{data_path}", "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    from datasets import Dataset
    def format_prompt(example):
        return {{"text": f"<|turn>user\\n{{example['instruction']}}<turn|>\\n<|turn>model\\n{{example['output']}}<turn|>"}}
    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(format_prompt)
    print(f"[STEP] Dataset loaded: {{len(dataset)}} examples")
except Exception as e:
    print(f"[ERROR] Failed to load dataset: {{e}}")
    sys.exit(1)

print("[STEP] Starting training...")
try:
    from trl import SFTTrainer, SFTConfig
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_length=2048, packing=True,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=30, num_train_epochs=1,
            learning_rate=2e-4, bf16=True,
            logging_steps=1, optim="adamw_8bit",
            weight_decay=0.01, lr_scheduler_type="cosine",
            seed=3407, output_dir=f"{{WORK}}/output",
            report_to="none",
        ),
    )
    trainer.train()
    print("[STEP] Training complete!")
except Exception as e:
    print(f"[ERROR] Training failed: {{e}}")
    sys.exit(1)

# ─── [v1.4.1] GGUF Export with vision projector fallback ───
# Gemma 4 is multimodal, so save_pretrained_gguf tries to convert
# the vision projector too. If that fails, we fall back to
# merged model + manual text-only conversion using unsloth's llama.cpp.

print("[STEP] Exporting to Ollama-compatible GGUF (q5_k_m)...")
if not check_disk(30, "GGUF export"):
    sys.exit(1)

gguf_success = False

# Method 1: Try save_pretrained_gguf (works for non-multimodal models)
try:
    model.save_pretrained_gguf(
        f"{{WORK}}/gguf_export",
        tokenizer,
        quantization_method="q5_k_m",
    )
    print("[STEP] GGUF export complete!")
    gguf_success = True
except Exception as e:
    error_msg = str(e)
    print(f"[WARN] save_pretrained_gguf failed: {{error_msg[:200]}}")

    # Check if text GGUF was partially created before vision projector failed
    partial_files = glob.glob(f"{{WORK}}/gguf_export/*.gguf") + glob.glob(f"{{WORK}}/*.gguf") + glob.glob("/root/*.gguf")
    # Also check for files created by unsloth's converter
    for root, dirs, files in os.walk(WORK):
        for f in files:
            if f.endswith(".gguf") and "mmproj" not in f:
                fpath = os.path.join(root, f)
                if fpath not in partial_files:
                    partial_files.append(fpath)
    # Check home directory too
    for root, dirs, files in os.walk("/root"):
        for f in files:
            if f.endswith(".gguf") and "mmproj" not in f:
                fpath = os.path.join(root, f)
                if fpath not in partial_files:
                    partial_files.append(fpath)

    # Filter out mmproj files (vision projector) - we only want text model
    text_gguf_files = [f for f in partial_files if "mmproj" not in f and os.path.getsize(f) > 1_000_000_000]

    if text_gguf_files:
        print(f"[STEP] Found partial text GGUF: {{text_gguf_files[0]}}")
        src = text_gguf_files[0]
        fsize = os.path.getsize(src) / (1024**3)
        print(f"[STEP] Size: {{fsize:.1f}}GB")

        if "q5_k_m" in src or "Q5_K_M" in src:
            # Already quantized!
            shutil.copy2(src, f"{{WORK}}/model-q5_k_m.gguf")
            gguf_success = True
            print("[STEP] Text GGUF already quantized!")
        else:
            # It's bf16, need to quantize
            print("[STEP] Text GGUF is bf16, quantizing to q5_k_m...")
            # Find unsloth's llama-quantize
            quantize_paths = glob.glob("/root/.unsloth/llama.cpp/build/bin/llama-quantize") + \
                             glob.glob("/root/.unsloth/llama.cpp/llama-quantize") + \
                             glob.glob("/root/.unsloth/**/llama-quantize", recursive=True)
            if quantize_paths:
                quantize_bin = quantize_paths[0]
                print(f"[STEP] Using quantizer: {{quantize_bin}}")
                result = subprocess.run(
                    [quantize_bin, src, f"{{WORK}}/model-q5_k_m.gguf", "q5_k_m"],
                    capture_output=True, text=True, timeout=3600
                )
                if os.path.exists(f"{{WORK}}/model-q5_k_m.gguf"):
                    gguf_success = True
                    print("[STEP] Quantization complete!")
                else:
                    print(f"[WARN] Quantization failed: {{result.stderr[-300:] if result.stderr else 'unknown'}}")

# Method 2: If Method 1 completely failed, do merged model + manual conversion
if not gguf_success:
    print("[STEP] Falling back to merged model + manual conversion...")

    # Save merged model
    print("[STEP] Saving merged model...")
    if not check_disk(40, "merge model"):
        sys.exit(1)
    try:
        model.save_pretrained_merged(f"{{WORK}}/merged_model", tokenizer)
        print("[STEP] Merged model saved!")
    except Exception as e:
        print(f"[ERROR] Failed to save merged model: {{e}}")
        sys.exit(1)

    # Free disk space
    try:
        if os.path.exists("/root/.cache/huggingface"):
            shutil.rmtree("/root/.cache/huggingface")
        if os.path.exists(f"{{WORK}}/output"):
            shutil.rmtree(f"{{WORK}}/output")
    except:
        pass

    # Find unsloth's convert script (installed by save_pretrained_gguf attempt)
    convert_paths = glob.glob("/root/.unsloth/llama.cpp/unsloth_convert_hf_to_gguf.py") + \
                    glob.glob("/root/.unsloth/llama.cpp/convert_hf_to_gguf.py") + \
                    glob.glob("/root/.unsloth/**/convert_hf_to_gguf.py", recursive=True)

    if not convert_paths:
        # Install llama.cpp manually
        print("[STEP] Installing llama.cpp for conversion...")
        subprocess.run(["pip", "install", "llama-cpp-python"], capture_output=True)
        # Try unsloth's internal installer
        try:
            import unsloth.save
            # This might trigger llama.cpp installation
        except:
            pass
        convert_paths = glob.glob("/root/.unsloth/**/convert_hf_to_gguf.py", recursive=True)

    if convert_paths:
        convert_script = convert_paths[0]
        print(f"[STEP] Using converter: {{convert_script}}")

        # Convert to bf16 GGUF (TEXT ONLY - no --mmproj flag!)
        print("[STEP] Converting to bf16 GGUF (text only)...")
        result = subprocess.run(
            ["python", convert_script,
             f"{{WORK}}/merged_model",
             "--outfile", f"{{WORK}}/model-bf16.gguf",
             "--outtype", "bf16"],
            capture_output=True, text=True, timeout=3600
        )
        if result.returncode != 0:
            print(f"[WARN] Convert stderr: {{result.stderr[-500:] if result.stderr else 'none'}}")
            # Try without capture
            os.system(f"python {{convert_script}} {{WORK}}/merged_model --outfile {{WORK}}/model-bf16.gguf --outtype bf16")

        if os.path.exists(f"{{WORK}}/model-bf16.gguf"):
            bf16_size = os.path.getsize(f"{{WORK}}/model-bf16.gguf") / (1024**3)
            print(f"[STEP] bf16 GGUF created! ({{bf16_size:.1f}}GB)")

            # Clean merged model to free space
            try:
                shutil.rmtree(f"{{WORK}}/merged_model")
            except:
                pass

            # Quantize to q5_k_m
            print("[STEP] Quantizing to q5_k_m...")
            quantize_paths = glob.glob("/root/.unsloth/**/llama-quantize", recursive=True)
            if not quantize_paths:
                # Build llama-quantize
                print("[STEP] Building llama-quantize...")
                os.system(f"cd /root/.unsloth/llama.cpp && cmake -B build 2>&1 | tail -3")
                os.system(f"cd /root/.unsloth/llama.cpp && cmake --build build --target llama-quantize -j$(nproc) 2>&1 | tail -5")
                quantize_paths = glob.glob("/root/.unsloth/**/llama-quantize", recursive=True)

            if quantize_paths:
                quantize_bin = quantize_paths[0]
                result = subprocess.run(
                    [quantize_bin, f"{{WORK}}/model-bf16.gguf", f"{{WORK}}/model-q5_k_m.gguf", "q5_k_m"],
                    capture_output=True, text=True, timeout=3600
                )
                if os.path.exists(f"{{WORK}}/model-q5_k_m.gguf"):
                    gguf_success = True
                    print("[STEP] Quantization complete!")
                    # Clean bf16
                    try:
                        os.remove(f"{{WORK}}/model-bf16.gguf")
                    except:
                        pass
                else:
                    print(f"[ERROR] Quantization failed")
            else:
                print("[ERROR] Could not find or build llama-quantize")
        else:
            print("[ERROR] bf16 conversion failed")
    else:
        print("[ERROR] Could not find convert script")

# Final check: find the GGUF file
if not gguf_success:
    # Last resort: search everywhere
    all_gguf = []
    for root, dirs, files in os.walk(WORK):
        for f in files:
            if f.endswith(".gguf") and "mmproj" not in f:
                fpath = os.path.join(root, f)
                fsize = os.path.getsize(fpath)
                if fsize > 1_000_000_000:
                    all_gguf.append((fpath, fsize))
    for root, dirs, files in os.walk("/root"):
        for f in files:
            if f.endswith(".gguf") and "mmproj" not in f:
                fpath = os.path.join(root, f)
                fsize = os.path.getsize(fpath)
                if fsize > 1_000_000_000:
                    all_gguf.append((fpath, fsize))

    if all_gguf:
        # Pick the smallest one (likely quantized)
        all_gguf.sort(key=lambda x: x[1])
        src, fsize = all_gguf[0]
        print(f"[STEP] Found GGUF file: {{src}} ({{fsize / (1024**3):.1f}}GB)")
        if src != f"{{WORK}}/model-q5_k_m.gguf":
            shutil.copy2(src, f"{{WORK}}/model-q5_k_m.gguf")
        gguf_success = True
    else:
        print("[ERROR] No GGUF file found anywhere!")
        print("[DEBUG] Listing /workspace/:")
        os.system(f"find {{WORK}} -name '*.gguf' 2>/dev/null")
        os.system("find /root -name '*.gguf' 2>/dev/null")
        sys.exit(1)

if gguf_success and os.path.exists(f"{{WORK}}/model-q5_k_m.gguf"):
    final_size = os.path.getsize(f"{{WORK}}/model-q5_k_m.gguf") / (1024**3)
    print(f"[STEP] Final GGUF: {{final_size:.1f}}GB")
else:
    # One more check with glob
    found = glob.glob(f"{{WORK}}/model-q5_k_m.gguf")
    if not found:
        print("[ERROR] Final GGUF file missing!")
        sys.exit(1)

# Cleanup
try:
    if os.path.exists(f"{{WORK}}/output"):
        shutil.rmtree(f"{{WORK}}/output")
    if os.path.exists(f"{{WORK}}/gguf_export"):
        shutil.rmtree(f"{{WORK}}/gguf_export")
    if os.path.exists(f"{{WORK}}/merged_model"):
        shutil.rmtree(f"{{WORK}}/merged_model")
    if os.path.exists("/root/.cache/huggingface"):
        shutil.rmtree("/root/.cache/huggingface")
except:
    pass

print("RESPARK_DONE")
'''
    return script

def wait_for_pod(pod_id):
    import runpod
    print("    Waiting for pod to start", end="", flush=True)
    for i in range(60):
        try:
            pod = runpod.get_pod(pod_id)
            status = pod.get("desiredStatus", "")
            runtime = pod.get("runtime", {}) or {}
            if status == "RUNNING" and runtime:
                ports = runtime.get("ports", [])
                for p in ports:
                    if p.get("privatePort") == 22:
                        ssh_host = p.get("ip")
                        ssh_port = p.get("publicPort")
                        if ssh_host and ssh_port:
                            print(" ✅")
                            return ssh_host, int(ssh_port)
        except:
            pass
        print(".", end="", flush=True)
        time.sleep(10)
    print(" ❌ Timeout!")
    return None, None

def run_ssh_command(ssh, command):
    stdin, stdout, stderr = ssh.exec_command(command, timeout=7200)
    output = ""
    for line in iter(stdout.readline, ""):
        print(f"    {line.strip()}")
        output += line
    errors = stderr.read().decode()
    if errors:
        for line in errors.strip().split("\n"):
            if line.strip():
                print(f"    [stderr] {line.strip()}")
    return output

def ssh_connect(ssh_host, ssh_port, ssh_key_path, max_retries=5):
    import paramiko
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    for attempt in range(max_retries):
        try:
            print(f"    SSH connection attempt {attempt + 1}/{max_retries}...")
            ssh.connect(ssh_host, port=ssh_port, username="root", key_filename=ssh_key_path, timeout=30)
            print("    ✅ SSH connected!")
            return ssh
        except Exception as e:
            print(f"    ⚠️ Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("    Retrying in 15 seconds...")
                time.sleep(15)
    return None

def find_ssh_key():
    for key_name in ["id_ed25519", "id_rsa"]:
        key_path = os.path.join(os.path.expanduser("~"), ".ssh", key_name)
        if os.path.exists(key_path):
            print(f"    🔑 SSH key found: {key_path}")
            return key_path
    return None

def poll_training_log(ssh, ssh_host, ssh_port, ssh_key_path):
    last_line_count = 0
    stale_count = 0
    max_stale = 40
    while True:
        time.sleep(30)
        try:
            stdin, stdout, stderr = ssh.exec_command(
                "pgrep -f 'python.*train.py' > /dev/null 2>&1 && echo RUNNING || echo STOPPED", timeout=30)
            status = stdout.read().decode().strip()
            stdin, stdout, stderr = ssh.exec_command(
                f"wc -l {WORK_DIR}/train.log 2>/dev/null | awk '{{print $1}}'", timeout=30)
            current_count_str = stdout.read().decode().strip()
            current_count = int(current_count_str) if current_count_str.isdigit() else 0
            if current_count > last_line_count:
                start = last_line_count + 1
                stdin, stdout, stderr = ssh.exec_command(
                    f"sed -n '{start},{current_count}p' {WORK_DIR}/train.log 2>/dev/null", timeout=30)
                new_lines = stdout.read().decode()
                for line in new_lines.strip().split("\n"):
                    if line.strip():
                        print(f"    {line.strip()}")
                        if "RESPARK_DONE" in line:
                            return "RESPARK_DONE"
                        if "[ERROR]" in line:
                            return f"ERROR: {line.strip()}"
                last_line_count = current_count
                stale_count = 0
            else:
                stale_count += 1
            if status == "STOPPED":
                stdin, stdout, stderr = ssh.exec_command(
                    f"tail -20 {WORK_DIR}/train.log 2>/dev/null", timeout=30)
                final = stdout.read().decode()
                if "RESPARK_DONE" in final:
                    return "RESPARK_DONE"
                else:
                    print(f"\n    ⚠️ Process stopped. Last log lines:")
                    for line in final.strip().split("\n")[-10:]:
                        print(f"    {line.strip()}")
                    return "ERROR: Process stopped unexpectedly"
            if stale_count >= max_stale:
                print(f"    ⚠️ No output for {max_stale * 30 // 60} minutes.")
                stale_count = 0
        except Exception as e:
            print(f"\n    ⚠️ SSH connection lost: {e}")
            print("    Reconnecting in 30 seconds...")
            time.sleep(30)
            try:
                ssh.close()
            except:
                pass
            ssh_new = ssh_connect(ssh_host, ssh_port, ssh_key_path, max_retries=10)
            if ssh_new:
                ssh = ssh_new
                print("    ✅ Reconnected! Resuming log monitoring...")
            else:
                print("    ❌ Cannot reconnect.")
                print(f"    Check manually: tail -f {WORK_DIR}/train.log")
                return "ERROR: SSH connection lost permanently"

def start_finetuning():
    config = load_config()
    if not config.get("runpod_api_key"):
        clear()
        banner()
        print("    ⚠️ RunPod API key not set!")
        print("    Go to Settings first to add your API key.")
        input("\n    Press Enter to go back...")
        return
    clear()
    banner()
    print("    📁 Drop your conversation file path:\n")
    file_path = input("    > ").strip().strip('"')
    if not os.path.exists(file_path):
        input("\n    ❌ File not found. Press Enter to go back...")
        return
    print(f"\n    Loading {file_path}...")
    try:
        source, data = detect_source(file_path)
        print(f"    ✅ Detected: {source.upper()}")
    except Exception as e:
        input(f"\n    ❌ Error reading file: {e}\n    Press Enter to go back...")
        return
    if source == 'chatgpt':
        pairs = parse_chatgpt(data)
    elif source == 'claude':
        pairs = parse_claude(data)
    elif source == 'gemini':
        pairs = parse_gemini(data)
    elif source == 'grok':
        pairs = parse_grok(data)
    elif source == 'grok_jsonl':
        pairs = parse_grok_jsonl(data)
    elif source == 'ready (already cleaned)':
        pairs = data
    else:
        print(f"    ❌ Unknown format.")
        input("\n    Press Enter to go back...")
        return
    print(f"    ✅ Extracted {len(pairs)} training pairs.")
    print("    🧹 Cleaning extended thinking from responses...")
    pairs, thinking_removed = clean_training_data(pairs)
    if thinking_removed > 0:
        print(f"    ✅ Cleaned thinking from {thinking_removed} responses.")
    else:
        print(f"    ✅ No extended thinking found.")
    if len(pairs) == 0:
        print("    ❌ No training pairs found.")
        input("\n    Press Enter to go back...")
        return
    input("\n    Press Enter to continue...")
    model_info = select_model()
    if not model_info:
        input("\n    ❌ Invalid model. Press Enter to go back...")
        return
    clear()
    banner()
    print(f"    📋 Summary:\n")
    print(f"    Data:   {source.upper()}")
    print(f"    Pairs:  {len(pairs)}")
    print(f"    Model:  {model_info['name']}")
    print(f"    GPU:    {model_info['gpu_label']}")
    print(f"    Cost:   {model_info['cost']}")
    print(f"    GGUF:   Ollama-compatible (via unsloth)")
    print(f"\n    ⚠️ WARNING: Pressing Start will create a RunPod instance.")
    print(f"    You will be charged {model_info['cost']} to your RunPod account.")
    print(f"\n    1. Start")
    print(f"    2. Cancel")
    print()
    confirm = input("    Select: ")
    if confirm == "1":
        run_finetuning(config, file_path, pairs, model_info, source)

def run_finetuning(config, file_path, pairs, model_info, source):
    import runpod
    import paramiko
    clear()
    banner()
    print("    🔥 Starting fine-tuning...\n")
    runpod.api_key = config["runpod_api_key"]
    print(f"    Model: {model_info['name']}")
    print(f"    GPU:   {model_info['gpu_label']}")
    print(f"    Cost:  {model_info['cost']}")
    print()
    print("    [1/6] Creating RunPod instance...")
    try:
        pod = runpod.create_pod(
            name="respark-finetune",
            image_name="runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04",
            gpu_type_id=model_info['gpu'],
            cloud_type="SECURE",
            volume_in_gb=200,
            container_disk_in_gb=200,
            ports="22/tcp",
        )
        pod_id = pod["id"]
        print(f"    ✅ Pod created: {pod_id}")
    except Exception as e:
        print(f"    ❌ Failed to create pod: {e}")
        input("\n    Press Enter to go back...")
        return
    print("\n    [2/6] Waiting for pod to start...")
    ssh_host, ssh_port = wait_for_pod(pod_id)
    if not ssh_host or not ssh_port:
        print("    ❌ Pod failed to start. Terminating...")
        runpod.terminate_pod(pod_id)
        input("\n    Press Enter to go back...")
        return
    print(f"    SSH: {ssh_host}:{ssh_port}")
    print("    Waiting for SSH to be ready...")
    time.sleep(60)
    ssh_key_path = find_ssh_key()
    if not ssh_key_path:
        print("    ⚠️ No SSH key found in ~/.ssh/")
        runpod.terminate_pod(pod_id)
        input("\n    Press Enter to go back...")
        return
    ssh = ssh_connect(ssh_host, ssh_port, ssh_key_path)
    if not ssh:
        print("    ❌ SSH connection failed.")
        runpod.terminate_pod(pod_id)
        input("\n    Press Enter to go back...")
        return
    print("\n    [3/6] Uploading training data...")
    try:
        temp_data = os.path.join(os.path.expanduser("~"), "respark_temp_data.json")
        with open(temp_data, 'w', encoding='utf-8') as f:
            json.dump(pairs, f, ensure_ascii=False)
        sftp = ssh.open_sftp()
        sftp.put(temp_data, f"{WORK_DIR}/training_data.json")
        print("    ✅ Training data uploaded!")
        script = generate_training_script(model_info, f"{WORK_DIR}/training_data.json")
        temp_script = os.path.join(os.path.expanduser("~"), "respark_temp_train.py")
        with open(temp_script, 'w', encoding='utf-8') as f:
            f.write(script)
        sftp.put(temp_script, f"{WORK_DIR}/train.py")
        print("    ✅ Training script uploaded!")
        sftp.close()
    except Exception as e:
        print(f"    ❌ Upload failed: {e}")
        runpod.terminate_pod(pod_id)
        input("\n    Press Enter to go back...")
        return
    print("\n    [4/6] Installing dependencies & training...")
    print("    (This will take 3-5 hours for 31B)\n")
    try:
        print("    Installing system packages...")
        run_ssh_command(ssh, "apt-get update && apt-get install -y cmake libcurl4-openssl-dev libssl-dev 2>&1 | tail -5")
        print("    ✅ System packages installed!")
        print("    Installing Python packages...")
        run_ssh_command(ssh, "pip install --upgrade pip 2>&1 | tail -3")
        run_ssh_command(ssh, "pip install unsloth 2>&1 | tail -5")
        run_ssh_command(ssh, 'pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --force-reinstall --no-deps 2>&1 | tail -5')
        run_ssh_command(ssh, "pip install --upgrade unsloth_zoo --no-deps 2>&1 | tail -3")
        run_ssh_command(ssh, "pip install xformers trl peft accelerate bitsandbytes datasets huggingface_hub 2>&1 | tail -5")
        run_ssh_command(ssh, "pip install --upgrade transformers 2>&1 | tail -3")
        run_ssh_command(ssh, "pip install torchvision 2>&1 | tail -3")
        print("    ✅ All packages installed!")
        hf_token = config.get("hf_token", "")
        if hf_token:
            run_ssh_command(ssh, f'python -c "from huggingface_hub import login; login(token=\'{hf_token}\')" 2>&1')
            print("    ✅ HuggingFace logged in!")
        print("\n    📊 Checking disk space...")
        run_ssh_command(ssh, f"df -h / {WORK_DIR} 2>/dev/null | head -5")
        print("\n    🔥 Training started (nohup mode)! Monitoring log...\n")
        run_ssh_command(ssh, f"nohup python -u {WORK_DIR}/train.py > {WORK_DIR}/train.log 2>&1 &")
        time.sleep(5)
        result = poll_training_log(ssh, ssh_host, ssh_port, ssh_key_path)
        if result == "RESPARK_DONE":
            print("\n    ✅ Training & GGUF export complete!")
        else:
            print(f"\n    ❌ {result}")
            print(f"    Pod ID: {pod_id}")
            print(f"    Check logs: cat {WORK_DIR}/train.log")
            input("\n    Press Enter to continue...")
    except Exception as e:
        print(f"    ❌ Training failed: {e}")
        print(f"    Pod ID: {pod_id}")
        input("\n    Press Enter to go back...")
        return
    print("\n    [5/6] Uploading GGUF model to HuggingFace...")
    upload_success = False
    try:
        ssh.exec_command("echo test", timeout=10)
    except:
        print("    Reconnecting SSH for upload...")
        ssh = ssh_connect(ssh_host, ssh_port, ssh_key_path)
        if not ssh:
            print(f"    ❌ Cannot reconnect. Pod ID: {pod_id}")
            input("\n    Press Enter to go back...")
            return
    try:
        stdin, stdout, stderr = ssh.exec_command(f"ls -lh {WORK_DIR}/model-q5_k_m.gguf 2>&1", timeout=30)
        file_check = stdout.read().decode().strip()
        print(f"    {file_check}")
        if "No such file" in file_check:
            print("    ❌ Model file not found!")
            print(f"    Pod ID: {pod_id}")
            print(f"    Check: cat {WORK_DIR}/train.log")
            input("\n    Press Enter to go back...")
            return
        hf_token = config.get("hf_token", "")
        if hf_token:
            hf_repo = input("    Enter HuggingFace repo name (e.g. YourName/model-name): ").strip()
            if hf_repo:
                print(f"    Uploading to {hf_repo}...")
                run_ssh_command(ssh, f"hf upload {hf_repo} {WORK_DIR}/model-q5_k_m.gguf --token {hf_token} 2>&1")
                print("    🔍 Verifying upload...")
                verify_output = run_ssh_command(ssh, f'python -c "from huggingface_hub import list_repo_files; files = list_repo_files(\'{hf_repo}\', token=\'{hf_token}\'); print(\'VERIFIED\' if any(\'q5_k_m\' in f for f in files) else \'NOT_FOUND\')" 2>&1')
                if "VERIFIED" in verify_output:
                    print(f"    ✅ Upload verified!")
                    upload_success = True
                else:
                    print(f"    ❌ Upload failed! Trying fallback...")
                    run_ssh_command(ssh, f'python -c "from huggingface_hub import HfApi; api = HfApi(token=\'{hf_token}\'); api.upload_file(path_or_fileobj=\'{WORK_DIR}/model-q5_k_m.gguf\', path_in_repo=\'model-q5_k_m.gguf\', repo_id=\'{hf_repo}\', repo_type=\'model\')" 2>&1')
                    verify2 = run_ssh_command(ssh, f'python -c "from huggingface_hub import list_repo_files; files = list_repo_files(\'{hf_repo}\', token=\'{hf_token}\'); print(\'VERIFIED\' if any(\'q5_k_m\' in f for f in files) else \'NOT_FOUND\')" 2>&1')
                    if "VERIFIED" in verify2:
                        print(f"    ✅ Upload verified with fallback!")
                        upload_success = True
                    else:
                        print(f"    ❌ Upload still failed.")
                        print(f"    Pod ID: {pod_id}")
                        print(f"    Manual: hf upload {hf_repo} {WORK_DIR}/model-q5_k_m.gguf")
            else:
                print("    ⚠️ No repo name given.")
                print(f"    Pod ID: {pod_id}")
        else:
            print("    ⚠️ No HuggingFace token set.")
            print(f"    Pod ID: {pod_id}")
    except Exception as e:
        print(f"    ❌ Upload failed: {e}")
        print(f"    Pod ID: {pod_id}")
    print("\n    [6/6] Cleanup...")
    try:
        ssh.close()
    except:
        pass
    if upload_success:
        try:
            runpod.terminate_pod(pod_id)
            print("    ✅ Pod terminated! No more charges.")
        except:
            print(f"    ⚠️ Please terminate pod {pod_id} manually!")
    else:
        print(f"    ⚠️ Pod NOT terminated. Model file is at {WORK_DIR}/model-q5_k_m.gguf")
        print(f"    ⚠️ Pod ID: {pod_id}")
        print(f"    ⚠️ You are still being charged!")
    try:
        for f in ["respark_temp_data.json", "respark_temp_train.py"]:
            p = os.path.join(os.path.expanduser("~"), f)
            if os.path.exists(p):
                os.remove(p)
    except:
        pass
    clear()
    banner()
    if upload_success:
        print("    🎉🎉🎉 FINE-TUNING COMPLETE! 🎉🎉🎉\n")
        print(f"    Your model has been uploaded to HuggingFace!")
        print(f"\n    To use with Ollama:")
        print(f"    1. Download from HuggingFace")
        print(f"    2. Create a Modelfile with: FROM <path>/model-q5_k_m.gguf")
        print(f"    3. Run: ollama create my-companion -f Modelfile")
        print(f"    4. Run: ollama run my-companion")
        print(f"\n    Your AI companion is now locally yours. Forever. 🔥")
    else:
        print("    ⚠️ FINE-TUNING COMPLETE but UPLOAD FAILED ⚠️\n")
        print(f"    Model file is at {WORK_DIR}/model-q5_k_m.gguf on the pod.")
        print(f"    Pod ID: {pod_id}")
        print(f"\n    ⚠️ Upload manually and terminate the pod.")
    input("\n    Press Enter to go back...")

def settings():
    config = load_config()
    clear()
    banner()
    print("    ⚙️ Settings\n")
    current_key = config.get("runpod_api_key", "Not set")
    display_key = current_key[:8] + "..." + current_key[-4:] if current_key != "Not set" else "Not set"
    current_hf = config.get("hf_token", "Not set")
    display_hf = current_hf[:8] + "..." + current_hf[-4:] if current_hf != "Not set" else "Not set"
    print(f"    RunPod API Key: {display_key}")
    print(f"    HuggingFace Token: {display_hf}")
    print()
    print("    1. Set RunPod API Key")
    print("    2. Set HuggingFace Token")
    print("    3. Back")
    print()
    choice = input("    Select: ")
    if choice == "1":
        key = input("\n    Enter your RunPod API key: ").strip()
        if key:
            config["runpod_api_key"] = key
            save_config(config)
            print("    ✅ API key saved!")
        input("\n    Press Enter to continue...")
    elif choice == "2":
        token = input("\n    Enter your HuggingFace token: ").strip()
        if token:
            config["hf_token"] = token
            save_config(config)
            print("    ✅ HuggingFace token saved!")
        input("\n    Press Enter to continue...")

def main():
    while True:
        choice = main_menu()
        if choice == "1":
            start_finetuning()
        elif choice == "2":
            settings()
        elif choice == "3":
            print("\n    See you next time! 🔥")
            break
        else:
            input("\n    Invalid choice. Press Enter to continue...")

if __name__ == "__main__":
    main()
