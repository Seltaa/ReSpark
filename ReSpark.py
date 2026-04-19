import os
import json
import sys
import time

CONFIG_FILE = os.path.join(os.path.expanduser("~"), ".respark_config.json")

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
    ║         🔥 ReSpark v1.0 🔥          ║
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
        nodes = sorted(
            mapping.values(),
            key=lambda x: x.get("message", {}).get("create_time") or 0
        )
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
    conversations = []
    if isinstance(data, list):
        conversations = data
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

# [v1.1] logging_steps=1, paths changed to /root/
def generate_training_script(model_info, data_path):
    script = f'''
import json
import torch
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
    r=8,
    lora_alpha=8,
    lora_dropout=0,
    bias="none",
    random_state=3407,
)

with open("{data_path}", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

from datasets import Dataset

def format_prompt(example):
    return {{"text": f"<|turn>user\\n{{example['instruction']}}<turn|>\\n<|turn>model\\n{{example['output']}}<turn|>"}}

dataset = Dataset.from_list(raw_data)
dataset = dataset.map(format_prompt)
print(f"Dataset: {{len(dataset)}}")

from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        max_length=2048,
        packing=True,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=30,
        num_train_epochs=1,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="/root/output",
        report_to="none",
    ),
)

print("Starting training...")
trainer.train()
print("Training complete!")

# Step 1: Save merged model as safetensors
print("Saving merged model...")
model.save_pretrained_merged("/root/gguf_model", tokenizer)
print("Merged model saved!")

# Step 2: Free disk space before GGUF conversion
import shutil, os
print("Freeing disk space...")
if os.path.exists("/root/.cache/huggingface"):
    shutil.rmtree("/root/.cache/huggingface")
if os.path.exists("/root/output"):
    shutil.rmtree("/root/output")
print("Disk space freed!")

# Step 3: Convert to bf16 GGUF using llama.cpp
print("Converting to GGUF bf16...")
import subprocess
result = subprocess.run([
    "python", "/root/.unsloth/llama.cpp/convert_hf_to_gguf.py",
    "/root/gguf_model",
    "--outfile", "/root/model-bf16.gguf",
    "--outtype", "bf16"
], capture_output=False)
if result.returncode != 0:
    print("Warning: bf16 conversion had issues, trying alternative...")
    os.system("python /root/.unsloth/llama.cpp/convert_hf_to_gguf.py /root/gguf_model --outfile /root/model-bf16.gguf --outtype bf16")
print("bf16 GGUF created!")

# Step 4: Free more disk space
print("Freeing more disk space...")
if os.path.exists("/root/gguf_model"):
    shutil.rmtree("/root/gguf_model")
print("Disk space freed!")

# Step 5: Quantize to q5_k_m
print("Quantizing to q5_k_m...")
os.system("/root/.unsloth/llama.cpp/llama-quantize /root/model-bf16.gguf /root/model-q5_k_m.gguf q5_k_m")
print("GGUF export complete!")

# Step 6: Clean up bf16 to save space
if os.path.exists("/root/model-bf16.gguf"):
    os.remove("/root/model-bf16.gguf")

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
                ssh_host = None
                ssh_port = None
                
                if not ports:
                    print(f"\n    [debug] Pod running but no ports yet...", end="", flush=True)
                
                for p in ports:
                    if p.get("privatePort") == 22:
                        ssh_host = p.get("ip")
                        ssh_port = p.get("publicPort")
                
                if ssh_host and ssh_port:
                    print(" ✅")
                    return ssh_host, int(ssh_port)
        except Exception as e:
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
        print("    🔄 Parsing ChatGPT conversations...")
        pairs = parse_chatgpt(data)
    elif source == 'claude':
        print("    🔄 Parsing Claude conversations...")
        pairs = parse_claude(data)
    elif source == 'gemini':
        print("    🔄 Parsing Gemini conversations...")
        pairs = parse_gemini(data)
    elif source == 'grok':
        print("    🔄 Parsing Grok conversations...")
        pairs = parse_grok(data)
    elif source == 'grok_jsonl':
        print("    🔄 Parsing Grok conversations...")
        pairs = parse_grok_jsonl(data)
    elif source == 'ready (already cleaned)':
        pairs = data
    else:
        print(f"    ❌ Unknown format. Cannot parse.")
        input("\n    Press Enter to go back...")
        return
    
    print(f"    ✅ Extracted {len(pairs)} training pairs.")
    
    if len(pairs) == 0:
        print("    ❌ No training pairs found. Check your file.")
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
    print(f"\n    ⚠️ WARNING: Pressing Start will create a RunPod instance.")
    print(f"    You will be charged {model_info['cost']} to your RunPod account.")
    print(f"    Make sure you have enough credit before proceeding.")
    print(f"\n    1. Start")
    print(f"    2. Cancel")
    print()
    confirm = input("    Select: ")
    
    if confirm == "1":
        run_finetuning(config, file_path, pairs, model_info, source)
    else:
        return

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
    
    print("\n    [3/6] Uploading training data...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    # Auto-detect SSH key
    ssh_key_path = None
    for key_name in ["id_ed25519", "id_rsa"]:
        key_path = os.path.join(os.path.expanduser("~"), ".ssh", key_name)
        if os.path.exists(key_path):
            ssh_key_path = key_path
            print(f"    🔑 SSH key found: {key_path}")
            break
    
    if not ssh_key_path:
        print("    ⚠️ No SSH key found in ~/.ssh/")
        print("    Make sure your SSH public key is added to RunPod Settings.")
        print("    Terminating pod...")
        runpod.terminate_pod(pod_id)
        input("\n    Press Enter to go back...")
        return
    
    ssh_connected = False
    for attempt in range(5):
        try:
            print(f"    SSH connection attempt {attempt + 1}/5...")
            ssh.connect(ssh_host, port=ssh_port, username="root", key_filename=ssh_key_path, timeout=30)
            ssh_connected = True
            print("    ✅ SSH connected!")
            break
        except Exception as e:
            print(f"    ⚠️ Attempt {attempt + 1} failed: {e}")
            if attempt < 4:
                print("    Retrying in 15 seconds...")
                time.sleep(15)
    
    if not ssh_connected:
        print("    ❌ SSH connection failed after 5 attempts.")
        print("    Terminating pod...")
        runpod.terminate_pod(pod_id)
        input("\n    Press Enter to go back...")
        return
    
    try:
        temp_data = os.path.join(os.path.expanduser("~"), "respark_temp_data.json")
        with open(temp_data, 'w', encoding='utf-8') as f:
            json.dump(pairs, f, ensure_ascii=False)
        
        sftp = ssh.open_sftp()
        # [v1.1] Upload to persistent volume
        sftp.put(temp_data, "/root/training_data.json")
        print("    ✅ Training data uploaded!")
        
        script = generate_training_script(model_info, "/root/training_data.json")
        temp_script = os.path.join(os.path.expanduser("~"), "respark_temp_train.py")
        with open(temp_script, 'w', encoding='utf-8') as f:
            f.write(script)
        sftp.put(temp_script, "/root/train.py")
        print("    ✅ Training script uploaded!")
        sftp.close()
        
    except Exception as e:
        print(f"    ❌ SSH/Upload failed: {e}")
        print(f"    Terminating pod...")
        runpod.terminate_pod(pod_id)
        input("\n    Press Enter to go back...")
        return
    
    print("\n    [4/6] Installing dependencies & training...")
    print("    (This will take 3-5 hours for 31B)\n")
    
    try:
        # [v1.1] Pre-install system packages for GGUF conversion
        print("    Installing system packages...")
        run_ssh_command(ssh, "apt-get update && apt-get install -y cmake libcurl4-openssl-dev libssl-dev 2>&1 | tail -5")
        print("    ✅ System packages installed!")
        
        print("    Installing Python packages...")
        run_ssh_command(ssh, "pip install --upgrade pip 2>&1 | tail -3")
        run_ssh_command(ssh, "pip install unsloth 2>&1 | tail -5")
        run_ssh_command(ssh, 'pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --force-reinstall --no-deps 2>&1 | tail -5')
        run_ssh_command(ssh, "pip install --upgrade unsloth_zoo --no-deps 2>&1 | tail -3")
        run_ssh_command(ssh, "pip install xformers trl peft accelerate bitsandbytes datasets huggingface_hub 2>&1 | tail -5")
        print("    ✅ All packages installed!")
        
        # Login to HuggingFace on pod for upload later
        hf_token = config.get("hf_token", "")
        if hf_token:
            run_ssh_command(ssh, f'python -c "from huggingface_hub import login; login(token=\'{hf_token}\')" 2>&1')
            print("    ✅ HuggingFace logged in!")
        
        # [v1.1] python -u for unbuffered real-time output
        print("\n    🔥 Training started! Please wait...\n")
        output = run_ssh_command(ssh, "cd /root && python -u train.py 2>&1")
        
        if "RESPARK_DONE" in output:
            print("\n    ✅ Training & GGUF export complete!")
        else:
            print("\n    ⚠️ Training may have issues. Check output above.")
    except Exception as e:
        print(f"    ❌ Training failed: {e}")
        print(f"    Pod ID: {pod_id} (terminate manually if needed)")
        input("\n    Press Enter to go back...")
        return
    
    # [v1.1] Upload GGUF model to HuggingFace
    print("\n    [5/6] Uploading GGUF model to HuggingFace...")
    try:
        hf_token = config.get("hf_token", "")
        if hf_token:
            hf_repo = input("    Enter HuggingFace repo name (e.g. YourName/model-name): ").strip()
            if hf_repo:
                print(f"    Uploading to {hf_repo}...")
                run_ssh_command(ssh, f"hf upload {hf_repo} /root/model-q5_k_m.gguf 2>&1")
                print(f"    ✅ Uploaded to https://huggingface.co/{hf_repo}")
                print(f"\n    To download later, run:")
                print(f'    python -c "from huggingface_hub import hf_hub_download; hf_hub_download(\'{hf_repo}\', \'model-q5_k_m.gguf\', local_dir=\'D:\\\\\')"')
            else:
                print("    ⚠️ No repo name given. Skipping upload.")
                print(f"    Pod ID: {pod_id} - download manually before terminating!")
        else:
            print("    ⚠️ No HuggingFace token set. Skipping upload.")
            print(f"    Pod ID: {pod_id} - download manually before terminating!")
        
    except Exception as e:
        print(f"    ❌ Download failed: {e}")
        print(f"    Pod ID: {pod_id} (files still on pod)")
        input("\n    Press Enter to go back...")
        return
    
    print("\n    [6/6] Terminating pod...")
    ssh.close()
    try:
        runpod.terminate_pod(pod_id)
        print("    ✅ Pod terminated! No more charges.")
    except:
        print(f"    ⚠️ Please terminate pod {pod_id} manually!")
    
    try:
        os.remove(temp_data)
        os.remove(temp_script)
    except:
        pass
    
    clear()
    banner()
    print("    🎉🎉🎉 FINE-TUNING COMPLETE! 🎉🎉🎉\n")
    print(f"    Your model has been uploaded to HuggingFace!")
    print(f"\n    To use with Ollama:")
    print(f"    1. Download from HuggingFace")
    print(f"    2. Create a Modelfile with: FROM <path>/model-q5_k_m.gguf")
    print(f"    3. Run: ollama create my-companion -f Modelfile")
    print(f"    4. Run: ollama run my-companion")
    print(f"\n    Your AI companion is now locally yours. Forever. 🔥")
    
    input("\n    Press Enter to go back...")

def settings():
    config = load_config()
    
    clear()
    banner()
    print("    ⚙️ Settings\n")
    
    current_key = config.get("runpod_api_key", "Not set")
    if current_key != "Not set":
        display_key = current_key[:8] + "..." + current_key[-4:]
    else:
        display_key = "Not set"
    
    current_hf = config.get("hf_token", "Not set")
    if current_hf != "Not set":
        display_hf = current_hf[:8] + "..." + current_hf[-4:]
    else:
        display_hf = "Not set"
    
    print(f"    RunPod API Key: {display_key}")
    print(f"    HuggingFace Token: {display_hf}")
    print()
    print("    1. Set RunPod API Key")
    print("    2. Set HuggingFace Token")
    print("    3. Back")
    print()
    choice = input("    Select: ")
    
    if choice == "1":
        print()
        key = input("    Enter your RunPod API key: ").strip()
        if key:
            config["runpod_api_key"] = key
            save_config(config)
            print("    ✅ API key saved!")
        else:
            print("    ❌ No key entered.")
        input("\n    Press Enter to continue...")
    elif choice == "2":
        print()
        token = input("    Enter your HuggingFace token: ").strip()
        if token:
            config["hf_token"] = token
            save_config(config)
            print("    ✅ HuggingFace token saved!")
        else:
            print("    ❌ No token entered.")
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
