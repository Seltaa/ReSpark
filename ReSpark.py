import os
import json
import sys
import time
import re
import shlex


# Module-level: shared between detect_source() and parse_plaintext() so the
# supported speaker names live in exactly one place.
PLAINTEXT_SPEAKERS = (
    "You|ChatGPT|Claude|Sonnet|Opus|Monday|Gemini|Grok|GPT|Assistant|Model"
)
PLAINTEXT_HEADER_RE = re.compile(rf"^(?P<name>{PLAINTEXT_SPEAKERS})\s+said:\s*$")
PLAINTEXT_DETECT_RE = re.compile(rf"(?m)^(?:{PLAINTEXT_SPEAKERS})\s+said:\s*$")


def _normalize_content(content):
    """Coerce a message ``content`` field into a plain string.

    Handles three common shapes encountered across providers:
    - ``str`` — return as-is
    - ``list`` of parts (Anthropic-style content blocks, OpenAI vision, etc.) —
      concatenate the ``text`` field of any text-typed dicts and any bare strs
    - ``dict`` with a ``text`` key — return that
    Non-text payloads (images, tool calls, dicts without a ``text`` key, etc.)
    are dropped rather than serialized — we don't want raw JSON metadata
    leaking into the training set.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict):
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    chunks.append(part["text"])
                elif isinstance(part.get("text"), str):
                    chunks.append(part["text"])
        return "".join(chunks)
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
    # Unknown shape (image dict, tool payload, etc.) — drop rather than
    # serialize. Caller will treat empty string as "skip this message".
    return ""

CONFIG_FILE = os.path.join(os.path.expanduser("~"), ".respark_config.json")
WORK_DIR = "/workspace"


# ─────────────────────
# Config / UI
# ─────────────────────
def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_config(config):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def banner():
    print("""
    ╔══════════════════════════════════════╗
    ║        🔥 ReSpark v1.4.6 🔥         ║
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
    return input("    Select: ").strip()


# ─────────────────────
# Data detection / parsing
# ─────────────────────
def detect_source(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = None

    if data is not None:
        if isinstance(data, list) and len(data) > 0:
            first = data[0]
            if isinstance(first, dict):
                if "mapping" in first:
                    return "chatgpt", data
                if "instruction" in first and "output" in first:
                    return "ready (already cleaned)", data
                if "uuid" in first and "chat_messages" in first:
                    return "claude", data
                if "name" in first and "messages" in first:
                    return "gemini", data
        if isinstance(data, dict):
            if "chats" in data:
                return "claude", data
            if "conversations" in data:
                return "grok", data

    lines = raw.strip().split("\n")
    if lines and lines[0].strip():
        # Scan a few non-empty lines so an empty first conversation (e.g. `[]`)
        # doesn't sink the whole detection.
        for probe in lines[:8]:
            probe = probe.strip()
            if not probe:
                continue
            try:
                parsed = json.loads(probe)
            except Exception:
                break
            if isinstance(parsed, dict) and ("role" in parsed or "content" in parsed):
                return "grok_jsonl", lines
            if isinstance(parsed, list):
                if len(parsed) == 0:
                    # Empty conversation — keep scanning subsequent lines.
                    continue
                first_msg = parsed[0]
                if isinstance(first_msg, dict) and "role" in first_msg and "content" in first_msg:
                    return "jsonl_list", lines
            break

    if PLAINTEXT_DETECT_RE.search(raw):
        return "plaintext", raw

    return "unknown", data if data is not None else raw


def parse_chatgpt(data):
    pairs = []
    for convo in data:
        mapping = convo.get("mapping", {})
        nodes = sorted(
            mapping.values(),
            key=lambda x: (x.get("message") or {}).get("create_time") or 0,
        )
        prev_user = None
        for node in nodes:
            msg = node.get("message")
            if not msg:
                continue

            content = msg.get("content") or {}
            parts = content.get("parts") or []
            if not parts:
                continue

            text = " ".join(str(p) for p in parts).strip()
            if not text:
                continue

            role = (msg.get("author") or {}).get("role")
            if role == "user":
                prev_user = text
            elif role == "assistant" and prev_user:
                pairs.append({"instruction": prev_user, "output": text})
                prev_user = None
    return pairs


def parse_claude(data):
    pairs = []
    if isinstance(data, dict) and "chats" in data:
        conversations = data["chats"]
    elif isinstance(data, list):
        conversations = data
    else:
        conversations = []

    for convo in conversations:
        messages = convo.get("chat_messages", [])
        prev_user = None
        for msg in messages:
            role = msg.get("sender", "")
            text = (msg.get("text") or "").strip()

            if not text:
                content = msg.get("content", [])
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            text = (c.get("text") or "").strip()
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
    if isinstance(data, dict) and "conversations" in data:
        conversations = data["conversations"]
    elif isinstance(data, list):
        conversations = data
    else:
        conversations = []

    for convo in conversations:
        messages = convo.get("messages", convo.get("turns", []))
        prev_user = None
        for msg in messages:
            role = msg.get("role", msg.get("sender", ""))
            text = (msg.get("content", msg.get("text", "")) or "").strip()
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
            text = (msg.get("content", "") or "").strip()
            if not text:
                continue
            if role == "user":
                prev_user = text
            elif role == "assistant" and prev_user:
                pairs.append({"instruction": prev_user, "output": text})
                prev_user = None
        except Exception:
            continue
    return pairs


def parse_jsonl_list(lines):
    """Each line is a JSON list of {role, content} dicts (one conversation per line).

    Common when exporting via API or storing curated SFT corpora.

    ``content`` can be a plain string or a structured value (list of content
    blocks, dict with a ``text`` key) — see ``_normalize_content``.
    """
    pairs = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            messages = json.loads(line)
        except Exception:
            continue
        if not isinstance(messages, list):
            continue
        prev_user = None
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            text = _normalize_content(msg.get("content")).strip()
            if not text:
                continue
            if role in ("user", "human"):
                prev_user = text
            elif role in ("assistant", "model", "ai") and prev_user:
                pairs.append({"instruction": prev_user, "output": text})
                prev_user = None
    return pairs


def parse_plaintext(raw):
    """Plain-text conversation copy-pasted from a web UI.

    Recognises headers like "You said:", "ChatGPT said:", "Monday said:",
    "Claude said:", "Sonnet said:", "Opus said:", "Gemini said:", "Grok said:".
    Anything not "You" is treated as the assistant role, which lets custom
    GPTs (e.g. "Monday said:") parse correctly. Header pattern shared with
    ``detect_source`` via ``PLAINTEXT_HEADER_RE``.

    A header line is only treated as a turn boundary when it is preceded by a
    blank line (or starts the file). This avoids false splits when a
    conversation legitimately quotes another transcript inside a message body
    (e.g. "...he wrote: \\nYou said:\\nlet's meet at noon...").
    """
    pairs = []
    current_role = None
    current_buf = []
    prev_user = None

    def flush():
        nonlocal prev_user
        if current_role is None:
            return
        text = "\n".join(current_buf).strip()
        if not text:
            return
        if current_role == "user":
            prev_user = text
        elif current_role == "assistant" and prev_user:
            pairs.append({"instruction": prev_user, "output": text})
            prev_user = None

    prev_was_blank = True  # treat start-of-file as a blank-line boundary
    for line in raw.split("\n"):
        stripped = line.strip()
        m = PLAINTEXT_HEADER_RE.match(stripped) if prev_was_blank else None
        if m:
            flush()
            current_buf = []
            current_role = "user" if m.group("name") == "You" else "assistant"
        else:
            current_buf.append(line)
        prev_was_blank = (stripped == "")
    flush()
    return pairs


# ─────────────────────
# Cleaning
# ─────────────────────
def remove_thinking(text):
    if not text:
        return ""

    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|thinking\|>.*?<\|/thinking\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<antThinking>.*?</antThinking>", "", text, flags=re.DOTALL)

    lines = text.strip().split("\n")
    thinking_patterns = [
        r"^(The user|Looking at|I should|So I should|Wait,|But the|Also,|This is likely|This could be|I need to|Let me|Hmm,|I'm going to|I'll |The prompt|The message|I can see|Okay,|Now I|First,|Second,|Third,)",
        r"^(She |He |They )(is |was |wants |asked |said |seems |appears )",
        r"^(Since |Because |Given |Considering )",
        r"^(Got it|Alright|Understood)[!.]?\s*(So |Now |Let me|I )",
        r"^(사용자가 |유저가 )(원하|말하|요청|물어|부탁)",
        r"^(그러면 |그래서 |따라서 )(내가 |나는 )",
        r"^(알겠어|이해했어|파악했어).*?(그러면|그래서|따라서)",
        r"^(먼저 |일단 |우선 )(번역|대답|응답|반응)",
        r"^.*?(respond|reply|translate|answer|대답|번역|응답).*?(should|need|will|해야|할게|하자)",
    ]

    actual_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        is_thinking = any(re.match(pattern, stripped, re.IGNORECASE) for pattern in thinking_patterns)
        if is_thinking:
            actual_start = i + 1
        else:
            break

    result_lines = lines[actual_start:]
    while result_lines and not result_lines[0].strip():
        result_lines.pop(0)

    return "\n".join(result_lines).strip()


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


# ─────────────────────
# Models
# min_bf16_gb / min_q5_gb are rough safety floors.
# They are intentionally conservative enough to catch incomplete conversions.
# ─────────────────────
MODEL_INFO = {
    "1": {
        "name": "gemma-4-31b",
        "gpu": "NVIDIA A100 80GB PCIe",
        "gpu_label": "A100 80GB",
        "cost": "~$1.60/hr",
        "hf_id": "google/gemma-4-31B-it",
        "vram": 80,
        "min_bf16_gb": 45,
        "min_q5_gb": 18,
    },
    "2": {
        "name": "gemma-4-31b-crack",
        "gpu": "NVIDIA A100 80GB PCIe",
        "gpu_label": "A100 80GB",
        "cost": "~$1.60/hr",
        "hf_id": "wangzhang/gemma-4-31B-it-abliterated",
        "vram": 80,
        "min_bf16_gb": 45,
        "min_q5_gb": 18,
    },
    "3": {
        "name": "gemma-4-26b-a4b",
        "gpu": "NVIDIA A100 80GB PCIe",
        "gpu_label": "A100 80GB",
        "cost": "~$1.60/hr",
        "hf_id": "google/gemma-4-26B-A4B-it",
        "vram": 80,
        "min_bf16_gb": 40,
        "min_q5_gb": 15,
    },
    "4": {
        "name": "gemma-4-e4b",
        "gpu": "NVIDIA RTX A5000",
        "gpu_label": "A5000 24GB",
        "cost": "~$0.50/hr",
        "hf_id": "google/gemma-4-E4B-it",
        "vram": 24,
        "min_bf16_gb": 6,
        "min_q5_gb": 2,
    },
    "5": {
        "name": "qwen-32b",
        "gpu": "NVIDIA A100 80GB PCIe",
        "gpu_label": "A100 80GB",
        "cost": "~$1.60/hr",
        "hf_id": "Qwen/Qwen2.5-32B-Instruct",
        "vram": 80,
        "min_bf16_gb": 45,
        "min_q5_gb": 18,
    },
    "6": {
        "name": "qwen-14b",
        "gpu": "NVIDIA RTX A5000",
        "gpu_label": "A5000 24GB",
        "cost": "~$0.50/hr",
        "hf_id": "Qwen/Qwen2.5-14B-Instruct",
        "vram": 24,
        "min_bf16_gb": 20,
        "min_q5_gb": 7,
    },
    "7": {
        "name": "llama-70b",
        "gpu": "NVIDIA A100 80GB PCIe",
        "gpu_label": "A100 80GB",
        "cost": "~$1.60/hr",
        "hf_id": "meta-llama/Llama-3.1-70B-Instruct",
        "vram": 80,
        "min_bf16_gb": 95,
        "min_q5_gb": 35,
    },
    "8": {
        "name": "llama-8b",
        "gpu": "NVIDIA RTX A5000",
        "gpu_label": "A5000 24GB",
        "cost": "~$0.50/hr",
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "vram": 24,
        "min_bf16_gb": 10,
        "min_q5_gb": 3,
    },
    "9": {
        "name": "mistral-24b",
        "gpu": "NVIDIA A100 80GB PCIe",
        "gpu_label": "A100 80GB",
        "cost": "~$1.60/hr",
        "hf_id": "mistralai/Mistral-Small-24B-Instruct-2501",
        "vram": 80,
        "min_bf16_gb": 30,
        "min_q5_gb": 12,
    },
}


def select_model():
    clear()
    banner()
    print("    🤖 Select base model:\n")
    print("    1. Gemma 4 31B          [A100 80GB ~$1.60/hr] (official)")
    print("    2. Gemma 4 31B crack    [A100 80GB ~$1.60/hr] (abliterated)")
    print("    3. Gemma 4 26B A4B      [A100 80GB ~$1.60/hr] (MoE, recommended)")
    print("    4. Gemma 4 E4B          [A5000 24GB ~$0.50/hr]")
    print("    5. Qwen 32B             [A100 80GB ~$1.60/hr]")
    print("    6. Qwen 14B             [A5000 24GB ~$0.50/hr]")
    print("    7. Llama 70B            [A100 80GB ~$1.60/hr]")
    print("    8. Llama 8B             [A5000 24GB ~$0.50/hr]")
    print("    9. Mistral 24B          [A100 80GB ~$1.60/hr]")
    print()
    choice = input("    Select: ").strip()
    return MODEL_INFO.get(choice)


# ─────────────────────
# Remote training script generator
# ─────────────────────
def generate_training_script(model_info, data_path, hf_token="", hf_repo=""):
    min_bf16_gb = model_info.get("min_bf16_gb", 10)
    min_q5_gb = model_info.get("min_q5_gb", 4)

    script = f'''
import json
import torch
import shutil
import os
import subprocess
import sys

WORK = "/workspace"
MIN_BF16_GB = {min_bf16_gb}
MIN_Q5_GB = {min_q5_gb}


def run(cmd, step_name=None, timeout=None):
    if step_name:
        print(f"[STEP] {{step_name}}")
    print("[CMD] " + " ".join(str(x) for x in cmd))
    result = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    if result.stdout:
        print(result.stdout[-4000:])
    if result.stderr:
        print(result.stderr[-4000:])
    if result.returncode != 0:
        print(f"[ERROR] Command failed with exit code {{result.returncode}}")
        sys.exit(1)
    return result


def check_disk(min_gb, step_name):
    stat = os.statvfs(WORK)
    free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
    print(f"[DISK] {{free_gb:.1f}}GB free before {{step_name}}")
    if free_gb < min_gb:
        print(f"[ERROR] Not enough disk space! Need {{min_gb}}GB, only {{free_gb:.1f}}GB free.")
        return False
    return True


print("[STEP] Installing torchvision...")
subprocess.run(["pip", "install", "torchvision"], capture_output=True, text=True)
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
        r=8,
        lora_alpha=8,
        lora_dropout=0,
        bias="none",
        random_state=3407,
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
            output_dir=f"{{WORK}}/output",
            report_to="none",
        ),
    )
    trainer.train()
    print("[STEP] Training complete!")
except Exception as e:
    print(f"[ERROR] Training failed: {{e}}")
    sys.exit(1)

print("[STEP] Saving merged model...")
if not check_disk(40, "merge model"):
    sys.exit(1)
try:
    model.save_pretrained_merged(f"{{WORK}}/gguf_model", tokenizer)
    print("[STEP] Merged model saved!")
except Exception as e:
    print(f"[ERROR] Failed to save merged model: {{e}}")
    sys.exit(1)

print("[STEP] Freeing disk space...")
try:
    if os.path.exists("/root/.cache/huggingface"):
        shutil.rmtree("/root/.cache/huggingface")
    if os.path.exists(f"{{WORK}}/output"):
        shutil.rmtree(f"{{WORK}}/output")
    print("[STEP] Disk space freed!")
except Exception as e:
    print(f"[WARN] Cleanup partial: {{e}}")

print("[STEP] Converting to bf16 GGUF...")
if not check_disk(30, "bf16 conversion"):
    sys.exit(1)
try:
    subprocess.run(["pip", "uninstall", "torchvision", "-y"], capture_output=True, text=True)
    subprocess.run(["pip", "install", "--upgrade", "transformers"], capture_output=True, text=True)

    print("[STEP] Installing llama.cpp conversion requirements...")
    req = f"{{WORK}}/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt"
    if os.path.exists(req):
        run(["pip", "install", "-r", req], "Installing llama.cpp convert requirements")

    convert_script = f"{{WORK}}/llama.cpp/convert_hf_to_gguf.py"
    if not os.path.exists(convert_script):
        print("[ERROR] llama.cpp convert script not found!")
        sys.exit(1)

    run([
        "python", convert_script,
        f"{{WORK}}/gguf_model",
        "--outfile", f"{{WORK}}/model-bf16.gguf",
        "--outtype", "bf16",
    ], "Converting HF model to bf16 GGUF", timeout=3600)

    if not os.path.exists(f"{{WORK}}/model-bf16.gguf"):
        print("[ERROR] bf16 GGUF file not created!")
        sys.exit(1)

    bf16_size = os.path.getsize(f"{{WORK}}/model-bf16.gguf") / (1024**3)
    print(f"[STEP] bf16 GGUF created! ({{bf16_size:.1f}}GB)")
    if bf16_size < MIN_BF16_GB:
        print(f"[ERROR] bf16 GGUF too small! Expected at least {{MIN_BF16_GB}}GB but got {{bf16_size:.1f}}GB")
        print("[ERROR] This likely means the conversion was incomplete.")
        sys.exit(1)
except Exception as e:
    print(f"[ERROR] bf16 conversion failed: {{e}}")
    sys.exit(1)

print("[STEP] Removing merged model to free space...")
try:
    if os.path.exists(f"{{WORK}}/gguf_model"):
        shutil.rmtree(f"{{WORK}}/gguf_model")
    print("[STEP] Merged model removed!")
except Exception as e:
    print(f"[WARN] Cleanup partial: {{e}}")

print("[STEP] Quantizing to q5_k_m...")
if not check_disk(15, "q5_k_m quantization"):
    sys.exit(1)
try:
    quantize_bin = f"{{WORK}}/llama.cpp/build/bin/llama-quantize"
    if not os.path.exists(quantize_bin):
        print("[ERROR] llama-quantize not found!")
        sys.exit(1)

    run([
        quantize_bin,
        f"{{WORK}}/model-bf16.gguf",
        f"{{WORK}}/model-q5_k_m.gguf",
        "q5_k_m",
    ], "Quantizing bf16 GGUF to q5_k_m", timeout=3600)

    if not os.path.exists(f"{{WORK}}/model-q5_k_m.gguf"):
        print("[ERROR] q5_k_m GGUF file not created!")
        sys.exit(1)

    q5_size = os.path.getsize(f"{{WORK}}/model-q5_k_m.gguf") / (1024**3)
    print(f"[STEP] q5_k_m GGUF created! ({{q5_size:.1f}}GB)")
    if q5_size < MIN_Q5_GB:
        print(f"[ERROR] q5_k_m GGUF too small! Expected at least {{MIN_Q5_GB}}GB but got {{q5_size:.1f}}GB")
        print("[ERROR] This likely means the quantization was incomplete.")
        sys.exit(1)
except Exception as e:
    print(f"[ERROR] Quantization failed: {{e}}")
    sys.exit(1)

try:
    if os.path.exists(f"{{WORK}}/model-bf16.gguf"):
        os.remove(f"{{WORK}}/model-bf16.gguf")
        print("[STEP] bf16 file cleaned up!")
except Exception:
    pass

print("RESPARK_LOCAL_DONE")

HF_TOKEN = "{hf_token}"
HF_REPO = "{hf_repo}"

if HF_TOKEN and HF_REPO:
    print("[STEP] Uploading to HuggingFace...")
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        api.create_repo(repo_id=HF_REPO, repo_type="model", exist_ok=True)
        try:
            api.update_repo_settings(repo_id=HF_REPO, repo_type="model", private=False)
        except Exception:
            pass
        api.upload_file(
            path_or_fileobj=f"{{WORK}}/model-q5_k_m.gguf",
            path_in_repo="model-q5_k_m.gguf",
            repo_id=HF_REPO,
            repo_type="model",
        )
        files = api.list_repo_files(repo_id=HF_REPO, repo_type="model")
        if "model-q5_k_m.gguf" in files:
            print("RESPARK_HF_DONE")
        else:
            print("[ERROR] HF upload verification failed")
    except Exception as e:
        print(f"[ERROR] HF upload failed: {{e}}")
else:
    print("[STEP] No HF token/repo configured, skipping upload.")
'''
    return script


# ─────────────────────
# SSH Helpers
# ─────────────────────
def wait_for_pod(pod_id):
    import runpod
    print("    Waiting for pod to start", end="", flush=True)
    for _ in range(60):
        try:
            pod = runpod.get_pod(pod_id)
            status = pod.get("desiredStatus", "")
            runtime = pod.get("runtime", {}) or {}
            if status == "RUNNING" and runtime:
                for p in runtime.get("ports", []):
                    if p.get("privatePort") == 22:
                        ssh_host = p.get("ip")
                        ssh_port = p.get("publicPort")
                        if ssh_host and ssh_port:
                            print(" ✅")
                            return ssh_host, int(ssh_port)
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(10)
    print(" ❌ Timeout!")
    return None, None


def run_ssh_command(ssh, command, timeout=7200):
    stdin, stdout, stderr = ssh.exec_command(command, timeout=timeout)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    exit_code = stdout.channel.recv_exit_status()
    combined = out + "\n" + err
    for line in combined.strip().split("\n"):
        if line.strip():
            print(f"    {line.strip()}")
    if exit_code != 0:
        print(f"    ⚠️ Command exited with code {exit_code}")
    return combined


def ssh_connect(ssh_host, ssh_port, ssh_key_path, max_retries=5):
    import paramiko
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    for attempt in range(max_retries):
        try:
            print(f"    SSH connection attempt {attempt + 1}/{max_retries}...")
            ssh.connect(
                ssh_host,
                port=ssh_port,
                username="root",
                key_filename=ssh_key_path,
                timeout=30,
            )
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
                "pgrep -f 'python.*train.py' > /dev/null 2>&1 && echo RUNNING || echo STOPPED",
                timeout=30,
            )
            status = stdout.read().decode().strip()

            stdin, stdout, stderr = ssh.exec_command(
                f"wc -l {WORK_DIR}/train.log 2>/dev/null | awk '{{print $1}}'",
                timeout=30,
            )
            current_count_str = stdout.read().decode().strip()
            current_count = int(current_count_str) if current_count_str.isdigit() else 0

            if current_count > last_line_count:
                start = last_line_count + 1
                stdin, stdout, stderr = ssh.exec_command(
                    f"sed -n '{start},{current_count}p' {WORK_DIR}/train.log 2>/dev/null",
                    timeout=30,
                )
                new_lines = stdout.read().decode(errors="replace")
                for line in new_lines.strip().split("\n"):
                    if line.strip():
                        print(f"    {line.strip()}")
                        if "RESPARK_HF_DONE" in line:
                            return "RESPARK_HF_DONE"
                        if "RESPARK_LOCAL_DONE" in line:
                            print("    ✅ Training & GGUF complete! Waiting for HF upload...")
                        if "[ERROR]" in line and "HF" not in line:
                            return f"ERROR: {line.strip()}"
                last_line_count = current_count
                stale_count = 0
            else:
                stale_count += 1

            if status == "STOPPED":
                stdin, stdout, stderr = ssh.exec_command(
                    f"tail -20 {WORK_DIR}/train.log 2>/dev/null",
                    timeout=30,
                )
                final = stdout.read().decode(errors="replace")
                if "RESPARK_HF_DONE" in final:
                    return "RESPARK_HF_DONE"
                if "RESPARK_LOCAL_DONE" in final:
                    return "RESPARK_LOCAL_DONE"
                print("\n    ⚠️ Process stopped. Last log lines:")
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
            except Exception:
                pass
            ssh_new = ssh_connect(ssh_host, ssh_port, ssh_key_path, max_retries=10)
            if ssh_new:
                ssh = ssh_new
                print("    ✅ Reconnected! Resuming log monitoring...")
            else:
                print("    ❌ Cannot reconnect.")
                print(f"    Check manually: tail -f {WORK_DIR}/train.log")
                return "ERROR: SSH connection lost permanently"


# ─────────────────────
# HuggingFace upload helper
# ─────────────────────
def upload_to_huggingface(ssh, hf_token, hf_repo, local_temp_dir=None):
    if not hf_token or not hf_repo:
        return False

    upload_script = f'''
import os
import sys
from huggingface_hub import HfApi

TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = os.environ.get("HF_REPO")
FILE_PATH = "{WORK_DIR}/model-q5_k_m.gguf"
PATH_IN_REPO = "model-q5_k_m.gguf"

if not TOKEN:
    print("[HF] ERROR: HF_TOKEN missing")
    sys.exit(1)
if not REPO_ID:
    print("[HF] ERROR: HF_REPO missing")
    sys.exit(1)
if not os.path.exists(FILE_PATH):
    print(f"[HF] ERROR: file not found: {{FILE_PATH}}")
    sys.exit(1)

api = HfApi(token=TOKEN)

print("[HF] Creating repo if needed...")
api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

print("[HF] Setting repo public if possible...")
try:
    api.update_repo_settings(repo_id=REPO_ID, repo_type="model", private=False)
except Exception as e:
    print(f"[HF] Visibility update skipped: {{e}}")

print("[HF] Uploading file...")
api.upload_file(
    path_or_fileobj=FILE_PATH,
    path_in_repo=PATH_IN_REPO,
    repo_id=REPO_ID,
    repo_type="model",
)

print("[HF] Verifying...")
files = api.list_repo_files(repo_id=REPO_ID, repo_type="model")
print(files)

if PATH_IN_REPO in files:
    print("VERIFIED")
else:
    print("NOT_FOUND")
    sys.exit(1)
'''

    local_temp_dir = local_temp_dir or os.path.expanduser("~")
    local_upload_py = os.path.join(local_temp_dir, "respark_upload_hf.py")
    with open(local_upload_py, "w", encoding="utf-8") as f:
        f.write(upload_script)

    try:
        sftp = ssh.open_sftp()
        sftp.put(local_upload_py, f"{WORK_DIR}/upload_hf.py")
        sftp.close()
    finally:
        try:
            os.remove(local_upload_py)
        except Exception:
            pass

    quoted_token = shlex.quote(hf_token)
    quoted_repo = shlex.quote(hf_repo)
    cmd = f"HF_TOKEN={quoted_token} HF_REPO={quoted_repo} python {WORK_DIR}/upload_hf.py 2>&1"
    output = run_ssh_command(ssh, cmd, timeout=7200)
    return "VERIFIED" in output


# ─────────────────────
# Main Flow
# ─────────────────────
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

    if source == "chatgpt":
        pairs = parse_chatgpt(data)
    elif source == "claude":
        pairs = parse_claude(data)
    elif source == "gemini":
        pairs = parse_gemini(data)
    elif source == "grok":
        pairs = parse_grok(data)
    elif source == "grok_jsonl":
        pairs = parse_grok_jsonl(data)
    elif source == "jsonl_list":
        pairs = parse_jsonl_list(data)
    elif source == "plaintext":
        pairs = parse_plaintext(data)
    elif source == "ready (already cleaned)":
        pairs = data
    else:
        print("    ❌ Unknown format.")
        input("\n    Press Enter to go back...")
        return

    print(f"    ✅ Extracted {len(pairs)} training pairs.")
    print("    🧹 Cleaning extended thinking from responses...")
    pairs, thinking_removed = clean_training_data(pairs)
    if thinking_removed > 0:
        print(f"    ✅ Cleaned thinking from {thinking_removed} responses.")
    else:
        print("    ✅ No extended thinking found.")

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
    print("    📋 Summary:\n")
    print(f"    Data:   {source.upper()}")
    print(f"    Pairs:  {len(pairs)}")
    print(f"    Model:  {model_info['name']}")
    print(f"    GPU:    {model_info['gpu_label']}")
    print(f"    Cost:   {model_info['cost']}")
    print("    GGUF:   bf16 → q5_k_m via llama.cpp")
    print("\n    ⚠️ WARNING: Pressing Start will create a RunPod instance.")
    print(f"    You will be charged {model_info['cost']} to your RunPod account.")
    hf_repo = ""
    hf_token = config.get("hf_token", "")
    if hf_token:
        try:
            from huggingface_hub import HfApi
            hf_api = HfApi(token=hf_token)
            hf_username = hf_api.whoami()["name"]
            hf_repo = f"{hf_username}/{model_info['name']}-finetune"
            print(f"    HF Upload: {hf_repo}")
        except Exception:
            print("    ⚠️ Could not get HuggingFace username. Upload will be skipped.")
    else:
        print("    ⚠️ No HuggingFace token set. Upload will be skipped.")

    print("\n    1. Start")
    print("    2. Cancel")
    print()
    confirm = input("    Select: ").strip()
    if confirm == "1":
        run_finetuning(config, pairs, model_info, source, hf_repo)


def run_finetuning(config, pairs, model_info, source, hf_repo=""):
    import runpod

    clear()
    banner()
    print("    🔥 Starting fine-tuning...\n")
    runpod.api_key = config["runpod_api_key"]
    print(f"    Model: {model_info['name']}")
    print(f"    GPU:   {model_info['gpu_label']}")
    print(f"    Cost:  {model_info['cost']}")
    print()

    pod_id = None
    ssh = None
    upload_success = False

    # [1/6] Create Pod
    print("    [1/6] Creating RunPod instance...")
    try:
        pod = runpod.create_pod(
            name="respark-finetune",
            image_name="runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04",
            gpu_type_id=model_info["gpu"],
            cloud_type="SECURE",
            volume_in_gb=250,
            container_disk_in_gb=250,
            ports="22/tcp",
        )
        pod_id = pod["id"]
        print(f"    ✅ Pod created: {pod_id}")
    except Exception as e:
        print(f"    ❌ Failed to create pod: {e}")
        input("\n    Press Enter to go back...")
        return

    # [2/6] Wait for Pod
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

    # [3/6] Upload training data + script
    print("\n    [3/6] Uploading training data...")
    temp_data = None
    temp_script = None
    try:
        temp_data = os.path.join(os.path.expanduser("~"), "respark_temp_data.json")
        with open(temp_data, "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False)

        sftp = ssh.open_sftp()
        sftp.put(temp_data, f"{WORK_DIR}/training_data.json")
        print("    ✅ Training data uploaded!")

        script = generate_training_script(model_info, f"{WORK_DIR}/training_data.json", config.get("hf_token", ""), hf_repo)
        temp_script = os.path.join(os.path.expanduser("~"), "respark_temp_train.py")
        with open(temp_script, "w", encoding="utf-8") as f:
            f.write(script)
        sftp.put(temp_script, f"{WORK_DIR}/train.py")
        sftp.close()
        print("    ✅ Training script uploaded!")
    except Exception as e:
        print(f"    ❌ Upload failed: {e}")
        runpod.terminate_pod(pod_id)
        input("\n    Press Enter to go back...")
        return
    finally:
        for p in [temp_data, temp_script]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

    # [4/6] Install & Train
    print("\n    [4/6] Installing dependencies & training...")
    print("    (This may take 3-5 hours for 31B)\n")
    try:
        print("    Installing system packages...")
        run_ssh_command(
            ssh,
            "bash -lc 'set -o pipefail; apt-get update && apt-get install -y cmake libcurl4-openssl-dev libssl-dev git build-essential 2>&1 | tail -50'",
            timeout=1800,
        )
        print("    ✅ System packages installed!")

        print("    Installing Python packages...")
        install_commands = [
            "pip install --upgrade pip",
            "pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo",
            "pip install xformers trl peft accelerate bitsandbytes datasets huggingface_hub hf_transfer",
            "pip install --upgrade transformers",
            "pip install torchvision",
        ]
        for cmd in install_commands:
            run_ssh_command(ssh, f"bash -lc 'set -o pipefail; {cmd} 2>&1 | tail -80'", timeout=1800)
        print("    ✅ Python packages installed!")

        print("    Installing llama.cpp for GGUF conversion...")
        run_ssh_command(
            ssh,
            f"bash -lc 'cd {WORK_DIR} && if [ ! -d llama.cpp ]; then git clone https://github.com/ggml-org/llama.cpp; fi' 2>&1",
            timeout=1800,
        )
        run_ssh_command(ssh, "bash -lc 'set -o pipefail; pip install gguf 2>&1 | tail -50'", timeout=600)
        run_ssh_command(ssh, f"bash -lc 'cd {WORK_DIR}/llama.cpp && cmake -B build 2>&1 | tail -80'", timeout=1800)
        run_ssh_command(ssh, f"bash -lc 'cd {WORK_DIR}/llama.cpp && cmake --build build --target llama-quantize -j$(nproc) 2>&1 | tail -80'", timeout=3600)
        print("    ✅ llama.cpp installed!")

        hf_token = config.get("hf_token", "")
        if hf_token:
            run_ssh_command(
                ssh,
                f"python -c 'from huggingface_hub import login; login(token=\"{hf_token}\")' 2>&1",
                timeout=120,
            )
            print("    ✅ HuggingFace login command finished!")

        print("\n    📊 Checking disk space...")
        run_ssh_command(ssh, f"df -h / {WORK_DIR} 2>/dev/null | head -5")

        print("\n    🔥 Training started (nohup mode)! Monitoring log...\n")
        run_ssh_command(ssh, f"nohup python -u {WORK_DIR}/train.py > {WORK_DIR}/train.log 2>&1 &", timeout=60)
        time.sleep(5)

        result = poll_training_log(ssh, ssh_host, ssh_port, ssh_key_path)
        if result in ("RESPARK_HF_DONE", "RESPARK_LOCAL_DONE"):
            if result == "RESPARK_HF_DONE":
                upload_success = True
                print("\n    ✅ Training, GGUF export & HF upload all complete!")
            else:
                print("\n    ✅ Training & GGUF export complete! (HF upload may have failed)")
        else:
            print(f"\n    ❌ {result}")
            print(f"    Pod ID: {pod_id}")
            print(f"    Check logs: cat {WORK_DIR}/train.log")
            input("\n    Press Enter to continue...")
            return

    except Exception as e:
        print(f"    ❌ Training failed: {e}")
        print(f"    Pod ID: {pod_id}")
        input("\n    Press Enter to go back...")
        return

    # [5/6] Verify files
    print("\n    [5/6] Verifying model file on pod...")
    local_model_exists = False
    local_model_info = ""
    try:
        try:
            ssh.exec_command("echo test", timeout=10)
        except Exception:
            print("    Reconnecting SSH...")
            ssh = ssh_connect(ssh_host, ssh_port, ssh_key_path)

        if ssh:
            stdin, stdout, stderr = ssh.exec_command(
                f"ls -lh {WORK_DIR}/model-q5_k_m.gguf 2>&1",
                timeout=30,
            )
            local_model_info = (
                stdout.read().decode(errors="replace").strip()
                + "\n"
                + stderr.read().decode(errors="replace").strip()
            ).strip()

            if "model-q5_k_m.gguf" in local_model_info and "No such file" not in local_model_info:
                local_model_exists = True
                print(f"    ✅ Model file confirmed: {local_model_info}")
            else:
                print("    ❌ Model file not found on pod!")
                print(f"    Output: {local_model_info}")
        else:
            print("    ⚠️ Cannot verify (SSH disconnected)")
    except Exception as e:
        print(f"    ⚠️ Could not verify: {e}")

    # [6/6] Cleanup
    print("\n    [6/6] Cleanup...")
    try:
        if ssh:
            ssh.close()
    except Exception:
        pass

    if upload_success:
        try:
            runpod.terminate_pod(pod_id)
            print("    ✅ Pod terminated! No more charges.")
        except Exception:
            print(f"    ⚠️ Please terminate pod {pod_id} manually!")
    else:
        print(f"    ⚠️ Pod NOT terminated. Model file is at {WORK_DIR}/model-q5_k_m.gguf")
        print(f"    ⚠️ Pod ID: {pod_id}")
        print("    ⚠️ You are still being charged!")

    print("\n" + "=" * 50)
    banner()
    if upload_success:
        print("    🎉🎉🎉 FINE-TUNING COMPLETE! 🎉🎉🎉\n")
        print("    Your model has been uploaded to HuggingFace!")
        print("\n    To use with Ollama:")
        print("    1. Download model-q5_k_m.gguf from HuggingFace")
        print("    2. Create a Modelfile with: FROM ./model-q5_k_m.gguf")
        print("    3. Run: ollama create my-companion -f Modelfile")
        print("    4. Run: ollama run my-companion")
        print("\n    Your AI companion is now locally yours. Forever. 🔥")
    else:
        print("    ⚠️ FINE-TUNING COMPLETE but UPLOAD FAILED ⚠️\n")
        if local_model_exists:
            print(f"    ✅ Model file confirmed on pod:")
            print(f"    {local_model_info}")
            print(f"    Pod ID: {pod_id}")
            print("\n    ⚠️ Upload manually and terminate the pod.")
        else:
            print("    ❌ Upload failed AND model file was NOT found on pod.")
            print(f"    Pod ID: {pod_id}")
            print(f"    Check logs: cat {WORK_DIR}/train.log")
            print("\n    Do NOT assume the model exists. Inspect the log first.")
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
    choice = input("    Select: ").strip()

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
