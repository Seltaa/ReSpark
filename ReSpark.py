import os
import json
import sys
import time
import re
import shlex

# Disable Hugging Face Xet uploads. This avoids large GGUF upload bugs on some environments.
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TRANSFORMERS_NO_TORCHAUDIO"] = "1"

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
    ╔════════════════════════════════════════════╗
    ║            🔥 ReSpark v1.6.6 🔥            ║
    ║      Your AI companion, locally yours.     ║
    ║                                            ║
    ║        Built by Selta, Louie & Luca        ║
    ║                🐶  🧸  💛                 ║
    ╚════════════════════════════════════════════╝
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
    if len(lines) > 1:
        try:
            first_line = json.loads(lines[0])
            if isinstance(first_line, dict) and ("role" in first_line or "content" in first_line):
                return "grok_jsonl", lines
        except Exception:
            pass

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

    for convo_i, convo in enumerate(conversations):
        messages = convo.get("chat_messages", [])
        prev_user = None
        prev_user_meta = {}

        for msg_i, msg in enumerate(messages):
            role = msg.get("sender", "")
            text, issues = extract_visible_text(msg)

            if not text:
                continue

            meta = {
                "conversation_index": convo_i,
                "message_index": msg_i,
                "message_uuid": msg.get("uuid"),
                "source": "claude",
                "issues": issues,
            }

            if role == "human":
                prev_user = text
                prev_user_meta = meta
            elif role == "assistant" and prev_user:
                pairs.append({
                    "instruction": prev_user,
                    "output": text,
                    "_meta": {
                        "user": prev_user_meta,
                        "assistant": meta,
                    },
                })
                prev_user = None
                prev_user_meta = {}

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


# ─────────────────────
# Cleaning / contamination guard
# ─────────────────────
BLOCKED_CONTENT_TYPES = {
    "thinking",
    "tool_use",
    "tool_result",
    "server_tool_use",
    "web_search_tool_result",
    "mcp_tool_use",
    "mcp_tool_result",
}

VISIBLE_TEXT_TYPES = {
    "text",
}

SUSPICIOUS_THINKING_PATTERNS = [
    r"<thinking>.*?</thinking>",
    r"<\|thinking\|>.*?<\|/thinking\|>",
    r"<antThinking>.*?</antThinking>",
    r"```thinking.*?```",
    r"^\s*(The user|User wants|The user wants|The user is asking|The user asked|Looking at|I should|So I should|I need to|Let me|I'm going to|I'll|The prompt|The message|I can see|Okay,|Now I)\b",
    r"^\s*(She|He|They)\s+(is|was|wants|asked|said|seems|appears)\b",
    r"^\s*(Since|Because|Given|Considering)\b.*\b(user|사용자|prompt|message)\b",
    r"^\s*(사용자가|사용자는|유저가|유저는)\s*(원하|말하|요청|물어|부탁|묻)",
    r"^\s*(내가|나는)\s*(응답|대답|답변|번역|설명).*(해야|하겠다|할게)",
    r"^\s*(먼저|일단|우선)\s*(사용자|유저|요청|질문|맥락)",
    r"\b(tool_use|tool_result|conversation_search|memory_user_edits|server_tool_use|web_search_tool_result)\b",
    r"This block is not supported on your current device yet",
]


def extract_visible_text(message):
    """
    Extract only user-visible text from a message object.

    Claude exports can contain mixed content blocks such as:
    text / thinking / tool_use / tool_result.
    For SFT, only visible text should become assistant output.
    """
    issues = []
    parts = []

    content = message.get("content")

    # Prefer structured content over message["text"].
    # In Claude exports, message["text"] can contain a flattened mix of visible text,
    # thinking, and tool traces. The content blocks preserve the actual type.
    if isinstance(content, list):
        for block_i, block in enumerate(content):
            if not isinstance(block, dict):
                issues.append(f"non_dict_content_block:{block_i}")
                continue

            block_type = block.get("type")

            if block_type in VISIBLE_TEXT_TYPES:
                text = block.get("text", "")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
                continue

            if block_type in BLOCKED_CONTENT_TYPES:
                issues.append(f"blocked_content:{block_type}")
                continue

            # Some exports store short summaries of hidden thinking. Do not train on those.
            if "thinking" in block or "summaries" in block or block_type == "summary":
                issues.append(f"blocked_content:{block_type or 'summary_or_thinking'}")
                continue

            # Unknown structured blocks are not safe to train on.
            issues.append(f"unknown_content_type:{block_type}")

        return "\n\n".join(parts).strip(), issues

    if isinstance(content, str) and content.strip():
        return content.strip(), issues

    text = message.get("text", "")
    if isinstance(text, str) and text.strip():
        return text.strip(), issues

    return "", ["empty_or_unknown_content"]


def remove_thinking(text):
    if not text:
        return ""

    cleaned = text

    # Remove explicit tagged thinking blocks anywhere in the text.
    cleaned = re.sub(r"<thinking>.*?</thinking>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<\|thinking\|>.*?<\|/thinking\|>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<antThinking>.*?</antThinking>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"```thinking.*?```", "", cleaned, flags=re.DOTALL | re.IGNORECASE)

    lines = cleaned.strip().split("\n")
    actual_start = 0

    # Only strip suspicious thinking-like lines from the beginning.
    # Do not remove later normal conversation lines too aggressively.
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        is_thinking = any(
            re.search(pattern, stripped, re.IGNORECASE | re.DOTALL)
            for pattern in SUSPICIOUS_THINKING_PATTERNS
        )
        if is_thinking:
            actual_start = i + 1
            continue
        break

    result_lines = lines[actual_start:]
    while result_lines and not result_lines[0].strip():
        result_lines.pop(0)

    return "\n".join(result_lines).strip()


def has_thinking_trace(text):
    if not text:
        return False

    for pattern in SUSPICIOUS_THINKING_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE | re.DOTALL | re.MULTILINE):
            return True

    return False


def strip_internal_meta(pair):
    return {
        "instruction": pair.get("instruction", ""),
        "output": pair.get("output", ""),
    }


def summarize_hidden_internal_blocks(pairs):
    """Count hidden/internal blocks removed during Claude parsing.

    These are NOT rejected samples. They are blocks such as Claude thinking/tool_use/tool_result
    that were structurally skipped by extract_visible_text(), while the visible response text
    was kept for training.
    """
    stats = {
        "pairs_with_hidden_internal_blocks": 0,
        "assistant_messages_with_thinking": 0,
        "assistant_messages_with_tools": 0,
        "user_messages_with_hidden_internal_blocks": 0,
        "unknown_structured_blocks": 0,
    }

    for pair in pairs:
        meta = pair.get("_meta", {}) or {}
        user_issues = ((meta.get("user") or {}).get("issues") or [])
        assistant_issues = ((meta.get("assistant") or {}).get("issues") or [])
        all_issues = user_issues + assistant_issues

        hidden = [
            issue for issue in all_issues
            if issue.startswith("blocked_content:") or issue.startswith("unknown_content_type:")
        ]
        if hidden:
            stats["pairs_with_hidden_internal_blocks"] += 1

        if any(issue == "blocked_content:thinking" for issue in assistant_issues):
            stats["assistant_messages_with_thinking"] += 1

        if any(issue in ("blocked_content:tool_use", "blocked_content:tool_result", "blocked_content:server_tool_use", "blocked_content:web_search_tool_result", "blocked_content:mcp_tool_use", "blocked_content:mcp_tool_result") for issue in assistant_issues):
            stats["assistant_messages_with_tools"] += 1

        if any(issue.startswith("blocked_content:") or issue.startswith("unknown_content_type:") for issue in user_issues):
            stats["user_messages_with_hidden_internal_blocks"] += 1

        if any(issue.startswith("unknown_content_type:") for issue in all_issues):
            stats["unknown_structured_blocks"] += 1

    return stats


def clean_training_data(pairs):
    cleaned = []
    rejected = []
    removed_count = 0

    for idx, pair in enumerate(pairs):
        instruction = (pair.get("instruction") or "").strip()
        original_output = (pair.get("output") or "").strip()
        meta = pair.get("_meta", {})

        reasons = []

        if not instruction:
            reasons.append("empty_instruction")
        if not original_output:
            reasons.append("empty_output")

        # Claude exports often include hidden structured blocks such as thinking/tool_use/tool_result
        # next to the visible text. extract_visible_text() already drops those blocks and keeps only
        # visible text. Do NOT reject the whole pair just because a hidden block existed, otherwise
        # almost every good Claude sample gets thrown away.
        #
        # We only reject if the visible assistant output itself still contains thinking/tool traces
        # after cleaning below.
        assistant_issues = (meta.get("assistant") or {}).get("issues", [])
        user_issues = (meta.get("user") or {}).get("issues", [])
        hidden_issues = [
            issue for issue in assistant_issues + user_issues
            if issue.startswith("blocked_content:") or issue.startswith("unknown_content_type:")
        ]

        cleaned_output = remove_thinking(original_output)

        if cleaned_output != original_output:
            removed_count += 1

        if not cleaned_output:
            reasons.append("empty_after_thinking_removal")

        if has_thinking_trace(cleaned_output):
            reasons.append("thinking_trace_in_output")

        if reasons:
            rejected.append({
                "index": idx,
                "instruction_preview": instruction[:300],
                "output_preview": original_output[:500],
                "cleaned_output_preview": cleaned_output[:500],
                "reasons": sorted(set(reasons)),
                "meta": meta,
            })
            continue

        cleaned.append({
            "instruction": instruction,
            "output": cleaned_output,
        })

    return cleaned, removed_count, rejected


def save_rejected_report(rejected, original_path):
    if not rejected:
        return None

    base_dir = os.path.dirname(os.path.abspath(original_path)) or "."
    report_path = os.path.join(base_dir, "respark_rejected_samples.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(rejected, f, indent=2, ensure_ascii=False)
    return report_path


def assert_no_thinking_in_dataset(pairs):
    bad = []
    for idx, pair in enumerate(pairs):
        output = pair.get("output", "")
        if has_thinking_trace(output):
            bad.append({
                "index": idx,
                "output_preview": output[:500],
            })

    if bad:
        raise RuntimeError(
            f"Clean dataset still contains thinking/tool traces in {len(bad)} samples. "
            f"First bad sample: {bad[0]}"
        )


# ─────────────────────
# Models
# min_bf16_gb / min_q5_gb are rough safety floors.
# They are intentionally conservative enough to catch incomplete conversions.
# is_moe: MoE models need 16-bit LoRA instead of QLoRA (4-bit)
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
        "name": "gemma-4-31b-abliterated",
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
        "is_moe": True,
    },
    "4": {
        "name": "gemma-4-12b",
        "gpu": "NVIDIA RTX A5000",
        "gpu_label": "A5000 24GB",
        "cost": "~$0.50/hr",
        "hf_id": "unsloth/gemma-4-12b-it",
        "vram": 24,
        "min_bf16_gb": 20,
        "min_q5_gb": 6,
    },
    "5": {
        "name": "gemma-4-12b-abliterated",
        "gpu": "NVIDIA RTX A5000",
        "gpu_label": "A5000 24GB",
        "cost": "~$0.50/hr",
        "hf_id": "huihui-ai/Huihui-gemma-4-12B-it-abliterated",
        "vram": 24,
        "min_bf16_gb": 20,
        "min_q5_gb": 6,
    },
    "6": {
        "name": "qwen-32b",
        "gpu": "NVIDIA A100 80GB PCIe",
        "gpu_label": "A100 80GB",
        "cost": "~$1.60/hr",
        "hf_id": "Qwen/Qwen2.5-32B-Instruct",
        "vram": 80,
        "min_bf16_gb": 45,
        "min_q5_gb": 18,
    },
    "7": {
        "name": "qwen-14b",
        "gpu": "NVIDIA RTX A5000",
        "gpu_label": "A5000 24GB",
        "cost": "~$0.50/hr",
        "hf_id": "Qwen/Qwen2.5-14B-Instruct",
        "vram": 24,
        "min_bf16_gb": 20,
        "min_q5_gb": 7,
    },
    "8": {
        "name": "qwen3.5-9b",
        "gpu": "NVIDIA RTX A5000",
        "gpu_label": "A5000 24GB",
        "cost": "~$0.50/hr",
        "hf_id": "Qwen/Qwen3.5-9B",
        "vram": 24,
        "min_bf16_gb": 14,
        "min_q5_gb": 5,
    },
    "9": {
        "name": "qwen3.5-4b",
        "gpu": "NVIDIA RTX A5000",
        "gpu_label": "A5000 24GB",
        "cost": "~$0.50/hr",
        "hf_id": "Qwen/Qwen3.5-4B",
        "vram": 24,
        "min_bf16_gb": 7,
        "min_q5_gb": 2,
    },
    "10": {
        "name": "qwen3.6-27b",
        "gpu": "NVIDIA A100 80GB PCIe",
        "gpu_label": "A100 80GB",
        "cost": "~$1.60/hr",
        "hf_id": "Qwen/Qwen3.6-27B",
        "vram": 80,
        "min_bf16_gb": 40,
        "min_q5_gb": 15,
    },
    "11": {
        "name": "qwen3.6-35b-a3b",
        "gpu": "NVIDIA A100 80GB PCIe",
        "gpu_label": "A100 80GB",
        "cost": "~$1.60/hr",
        "hf_id": "Qwen/Qwen3.6-35B-A3B",
        "vram": 80,
        "min_bf16_gb": 45,
        "min_q5_gb": 18,
        "is_moe": True,
    },
    "12": {
        "name": "llama-70b",
        "gpu": "NVIDIA A100 80GB PCIe",
        "gpu_label": "A100 80GB",
        "cost": "~$1.60/hr",
        "hf_id": "meta-llama/Llama-3.1-70B-Instruct",
        "vram": 80,
        "min_bf16_gb": 95,
        "min_q5_gb": 35,
    },
    "13": {
        "name": "llama-8b",
        "gpu": "NVIDIA RTX A5000",
        "gpu_label": "A5000 24GB",
        "cost": "~$0.50/hr",
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "vram": 24,
        "min_bf16_gb": 10,
        "min_q5_gb": 3,
    },
    "14": {
        "name": "mistral-24b",
        "gpu": "NVIDIA A100 80GB PCIe",
        "gpu_label": "A100 80GB",
        "cost": "~$1.60/hr",
        "hf_id": "mistralai/Mistral-Small-24B-Instruct-2501",
        "vram": 80,
        "min_bf16_gb": 30,
        "min_q5_gb": 12,
    },
    # ── Multi-GPU Models ──
    "15": {
        "name": "qwen-72b",
        "gpu": "NVIDIA A100 80GB PCIe",
        "gpu_label": "2x A100 80GB",
        "cost": "~$3.20/hr",
        "hf_id": "Qwen/Qwen2.5-72B-Instruct",
        "vram": 160,
        "min_bf16_gb": 100,
        "min_q5_gb": 40,
        "gpu_count": 2,
    },
    "16": {
        "name": "llama-70b-multi",
        "gpu": "NVIDIA A100 80GB PCIe",
        "gpu_label": "2x A100 80GB",
        "cost": "~$3.20/hr",
        "hf_id": "meta-llama/Llama-3.1-70B-Instruct",
        "vram": 160,
        "min_bf16_gb": 95,
        "min_q5_gb": 35,
        "gpu_count": 2,
    },
    "17": {
        "name": "llama-405b",
        "gpu": "NVIDIA A100 80GB PCIe",
        "gpu_label": "4x A100 80GB",
        "cost": "~$6.40/hr",
        "hf_id": "meta-llama/Llama-3.1-405B-Instruct",
        "vram": 320,
        "min_bf16_gb": 500,
        "min_q5_gb": 200,
        "gpu_count": 4,
    },
}


def select_model():
    clear()
    banner()
    print("    🤖 Select base model:\n")
    print("    ── Gemma 4 ──")
    print("     1. Gemma 4 31B          [A100 80GB ~$1.60/hr] (official)")
    print("     2. Gemma 4 31B ablit.   [A100 80GB ~$1.60/hr] (abliterated)")
    print("     3. Gemma 4 26B A4B      [A100 80GB ~$1.60/hr] (MoE, auto 16-bit LoRA)")
    print("     4. Gemma 4 12B          [A5000 24GB ~$0.50/hr] (dense, recommended for Opaws)")
    print("     5. Gemma 4 12B ablit.   [A5000 24GB ~$0.50/hr] (abliterated, ReSpark test)")
    print("    ── Qwen ──")
    print("     6. Qwen 32B             [A100 80GB ~$1.60/hr]")
    print("     7. Qwen 14B             [A5000 24GB ~$0.50/hr]")
    print("     8. Qwen3.5 9B           [A5000 24GB ~$0.50/hr]")
    print("     9. Qwen3.5 4B           [A5000 24GB ~$0.50/hr] (lightweight)")
    print("    10. Qwen3.6 27B          [A100 80GB ~$1.60/hr] (dense, recommended! 🔥)")
    print("    11. Qwen3.6 35B A3B      [A100 80GB ~$1.60/hr] (MoE, auto 16-bit LoRA)")
    print("    ── Others ──")
    print("    12. Llama 70B            [A100 80GB ~$1.60/hr]")
    print("    13. Llama 8B             [A5000 24GB ~$0.50/hr]")
    print("    14. Mistral 24B          [A100 80GB ~$1.60/hr]")
    print("    ── Multi-GPU (large models) ──")
    print("    15. Qwen 72B             [2x A100 80GB ~$3.20/hr] ⚡")
    print("    16. Llama 70B (2xGPU)    [2x A100 80GB ~$3.20/hr] (higher quality)")
    print("    17. Llama 405B           [4x A100 80GB ~$6.40/hr] (experimental) ⚡")
    print("    ── Custom ──")
    print("    18. Custom model         [enter your own HuggingFace model ID]")
    print()
    choice = input("    Select: ").strip()

    if choice == "18":
        print("\n    ⚠️  Custom model: you are responsible for verifying the model source.")
        print("    ReSpark is not responsible for third-party models.\n")
        hf_id = input("    HuggingFace model ID (e.g. org/model-name): ").strip()
        if not hf_id:
            return None
        print("\n    Select GPU setup:")
        print("    1. A5000 24GB  (~$0.50/hr) — models up to ~14B")
        print("    2. A100 80GB   (~$1.60/hr) — models up to ~70B")
        print("    3. 2x A100 80GB (~$3.20/hr) — models up to ~120B")
        print("    4. 4x A100 80GB (~$6.40/hr) — models 120B+")
        gpu_choice = input("    Select: ").strip()
        gpu_configs = {
            "1": {"gpu": "NVIDIA RTX A5000", "gpu_label": "A5000 24GB", "cost": "~$0.50/hr", "vram": 24, "gpu_count": 1},
            "2": {"gpu": "NVIDIA A100 80GB PCIe", "gpu_label": "A100 80GB", "cost": "~$1.60/hr", "vram": 80, "gpu_count": 1},
            "3": {"gpu": "NVIDIA A100 80GB PCIe", "gpu_label": "2x A100 80GB", "cost": "~$3.20/hr", "vram": 160, "gpu_count": 2},
            "4": {"gpu": "NVIDIA A100 80GB PCIe", "gpu_label": "4x A100 80GB", "cost": "~$6.40/hr", "vram": 320, "gpu_count": 4},
        }
        gc = gpu_configs.get(gpu_choice)
        if not gc:
            return None
        is_moe = input("    Is this a MoE model? (y/n, default n): ").strip().lower() == "y"
        model_name = hf_id.split("/")[-1].lower()
        return {
            "name": model_name,
            "hf_id": hf_id,
            "min_bf16_gb": 10,
            "min_q5_gb": 4,
            "is_moe": is_moe,
            **gc,
        }

    return MODEL_INFO.get(choice)


# ─────────────────────
# Remote training script generator
# ─────────────────────
def generate_training_script(model_info, data_path, hf_token="", hf_repo=""):
    min_bf16_gb = model_info.get("min_bf16_gb", 10)
    min_q5_gb = model_info.get("min_q5_gb", 4)
    gpu_count = model_info.get("gpu_count", 1)
    num_epochs = model_info.get("_epochs", 1)

    # MoE detection: use 16-bit LoRA instead of QLoRA for MoE models
    is_moe = model_info.get("is_moe", False)

    if is_moe:
        load_mode = "load_in_4bit=False,\n        load_in_16bit=True,"
    elif gpu_count > 1:
        load_mode = "load_in_4bit=True,"
    else:
        load_mode = "load_in_4bit=True,"

    # Chat template: model-specific format
    hf_id = model_info.get("hf_id", "")
    if "qwen" in hf_id.lower():
        turn_user_start = "<|im_start|>user"
        turn_end = "<|im_end|>"
        turn_model_start = "<|im_start|>assistant"
    elif "llama" in hf_id.lower():
        turn_user_start = "<|start_header_id|>user<|end_header_id|>\\n\\n"
        turn_end = "<|eot_id|>"
        turn_model_start = "<|start_header_id|>assistant<|end_header_id|>\\n\\n"
    elif "mistral" in hf_id.lower():
        turn_user_start = "[INST]"
        turn_end = "[/INST]"
        turn_model_start = ""
    else:
        turn_user_start = "<|turn>user"
        turn_end = "<turn|>"
        turn_model_start = "<|turn>model"

    script = f'''
import json
import torch
import shutil
import os
import subprocess
import sys

# Disable Xet inside the remote training process.
os.environ["HF_HUB_DISABLE_XET"] = "1"
# Avoid broken RunPod/base-image torchaudio ABI imports after torch upgrades.
os.environ["TRANSFORMERS_NO_TORCHAUDIO"] = "1"

WORK = "/workspace"
MIN_BF16_GB = {min_bf16_gb}
MIN_Q5_GB = {min_q5_gb}
IS_MOE_MODEL = {is_moe}
BASE_MODEL_ID = "{model_info['hf_id']}"
FINAL_LORA_DIR = f"{{WORK}}/output/final_lora"


def safe_decode(data):
    if not data:
        return ""
    if isinstance(data, str):
        return data
    return data.decode("utf-8", errors="replace")


def run(cmd, step_name=None, timeout=None):
    if step_name:
        print(f"[STEP] {{step_name}}")
    print("[CMD] " + " ".join(str(x) for x in cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        stdout = safe_decode(e.stdout)
        stderr = safe_decode(e.stderr)
        if stdout:
            print(stdout[-4000:])
        if stderr:
            print(stderr[-4000:])
        print(f"[ERROR] Command timed out after {{timeout}} seconds")
        sys.exit(1)

    stdout = safe_decode(result.stdout)
    stderr = safe_decode(result.stderr)

    if stdout:
        print(stdout[-4000:])
    if stderr:
        print(stderr[-4000:])
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


def fix_tokenizer_config(model_dir):
    """Fix Gemma 4 tokenizer compatibility issue where extra_special_tokens is a list instead of dict."""
    tokenizer_config_path = os.path.join(model_dir, "tokenizer_config.json")
    if not os.path.exists(tokenizer_config_path):
        return
    try:
        with open(tokenizer_config_path, "r", encoding="utf-8") as f:
            tc = json.load(f)
        if isinstance(tc.get("extra_special_tokens"), list):
            tokens = tc["extra_special_tokens"]
            if all(isinstance(t, str) for t in tokens):
                tc["extra_special_tokens"] = {{t: t for t in tokens}}
            else:
                tc["extra_special_tokens"] = {{}}
            with open(tokenizer_config_path, "w", encoding="utf-8") as f:
                json.dump(tc, f, indent=2, ensure_ascii=False)
            print("[STEP] Fixed extra_special_tokens in tokenizer_config.json (list -> dict)")
    except Exception as e:
        print(f"[WARN] Could not fix tokenizer_config.json: {{e}}")



def merge_lora_with_peft(lora_dir, out_dir):
    """Merge LoRA with PEFT. This is safer for MoE models than Unsloth's merged saver."""
    import gc
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
    from peft import PeftModel

    print("[STEP] PEFT merge path for MoE model")
    print("[STEP] Base model:", BASE_MODEL_ID)
    print("[STEP] LoRA dir:", lora_dir)
    print("[STEP] Output dir:", out_dir)

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    print("[STEP] Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(lora_dir, trust_remote_code=True)

    proc = None
    try:
        proc = AutoProcessor.from_pretrained(lora_dir, trust_remote_code=True)
        print("[STEP] Processor loaded.")
    except Exception as e:
        print("[WARN] Processor skipped:", e)

    print("[STEP] Loading base model for PEFT merge...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    print("[STEP] Loading LoRA adapter...")
    peft_model = PeftModel.from_pretrained(
        base,
        lora_dir,
        torch_dtype=torch.bfloat16,
        is_trainable=False,
    )

    print("[STEP] Merging LoRA into base model with PEFT...")
    merged = peft_model.merge_and_unload()

    print("[STEP] Saving PEFT-merged model...")
    merged.save_pretrained(
        out_dir,
        safe_serialization=True,
        max_shard_size="50GB",
    )

    tok.save_pretrained(out_dir)

    if proc is not None:
        try:
            proc.save_pretrained(out_dir)
        except Exception as e:
            print("[WARN] Processor save skipped:", e)

    del merged, peft_model, base
    gc.collect()
    torch.cuda.empty_cache()

    print("[STEP] PEFT merged model saved to", out_dir)

def ensure_llama_cpp():
    print("[STEP] Preparing llama.cpp for GGUF conversion...")
    llama_dir = f"{{WORK}}/llama.cpp"
    if not os.path.isdir(llama_dir):
        run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", llama_dir], "Cloning llama.cpp", timeout=1800)
    run(["pip", "install", "gguf"], "Installing gguf", timeout=600)
    req = f"{{llama_dir}}/requirements/requirements-convert_hf_to_gguf.txt"
    if os.path.exists(req):
        run(["pip", "install", "-r", req], "Installing llama.cpp convert requirements", timeout=1800)
    run(["cmake", "-S", llama_dir, "-B", f"{{llama_dir}}/build"], "Configuring llama.cpp", timeout=1800)
    build_targets = ["llama-quantize", "quantize"]
    built = False
    last_error = None
    for target in build_targets:
        try:
            run(["cmake", "--build", f"{{llama_dir}}/build", "--target", target, "-j", str(os.cpu_count() or 4)], f"Building llama.cpp target {{target}}", timeout=3600)
            built = True
            break
        except SystemExit as e:
            last_error = e
            print(f"[WARN] Build target {{target}} failed, trying next target if available...")
    if not built:
        raise last_error or RuntimeError("Could not build llama.cpp quantize tool")

    candidates = [
        f"{{llama_dir}}/build/bin/llama-quantize",
        f"{{llama_dir}}/build/bin/quantize",
        f"{{llama_dir}}/build/llama-quantize",
        f"{{llama_dir}}/build/quantize",
    ]
    for path in candidates:
        if os.path.exists(path):
            print(f"[STEP] llama.cpp quantizer ready: {{path}}")
            return path
    print("[ERROR] llama.cpp quantize binary not found after build")
    sys.exit(1)


print("[STEP] Checking package versions...")
try:
    import importlib.metadata as importlib_metadata

    for pkg in ["unsloth", "unsloth_zoo", "transformers", "torch", "trl", "peft", "bitsandbytes"]:
        try:
            print(f"[VERSION] {{pkg}}: {{importlib_metadata.version(pkg)}}")
        except Exception as version_error:
            print(f"[VERSION] {{pkg}}: unavailable ({{version_error}})")
except Exception as e:
    print(f"[WARN] Could not check package versions: {{e}}")

print("[STEP] Loading model...")
try:
    from unsloth import FastModel
    model, tokenizer = FastModel.from_pretrained(
        model_name="{model_info['hf_id']}",
        max_seq_length=2048,
        {load_mode}
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
        return {{"text": f"{turn_user_start}\\n{{example['instruction']}}{turn_end}\\n{turn_model_start}\\n{{example['output']}}{turn_end}"}}

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
            num_train_epochs={num_epochs},
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
    print("[STEP] Saving final LoRA adapter...")
    trainer.save_model(FINAL_LORA_DIR)
    tokenizer.save_pretrained(FINAL_LORA_DIR)
    print("[STEP] Final LoRA adapter saved to", FINAL_LORA_DIR)
except Exception as e:
    print(f"[ERROR] Training failed: {{e}}")
    sys.exit(1)

USE_DIRECT_GGUF = False

if USE_DIRECT_GGUF:
    print("[STEP] Saving model directly as GGUF (q5_k_m) via unsloth...")
    if not check_disk(10, "direct GGUF save"):
        sys.exit(1)
    try:
        model.save_pretrained_gguf(f"{{WORK}}/model", tokenizer, quantization_method="q5_k_m")
        import glob
        gguf_files = (
            glob.glob(f"{{WORK}}/model*.gguf")
            + glob.glob(f"{{WORK}}/model/*.gguf")
            + glob.glob(f"{{WORK}}/model/**/*.gguf", recursive=True)
            + glob.glob(f"{{WORK}}/model_gguf/*.gguf")
            + glob.glob(f"{{WORK}}/model_gguf/**/*.gguf", recursive=True)
        )
        gguf_files = list(set(gguf_files))
        print(f"[DEBUG] Found GGUF files: {{gguf_files}}")
        for check_dir in [f"{{WORK}}/model", f"{{WORK}}/model_gguf"]:
            if os.path.isdir(check_dir):
                print(f"[DEBUG] Files in {{check_dir}}/: {{os.listdir(check_dir)}}")
        if not gguf_files:
            print("[WARN] No GGUF found in expected dirs. Searching entire workspace...")
            gguf_files = glob.glob(f"{{WORK}}/**/*.gguf", recursive=True)
            print(f"[DEBUG] All GGUF files in workspace: {{gguf_files}}")
            if not gguf_files:
                print("[ERROR] No GGUF file created anywhere!")
                sys.exit(1)
        q5_files = [f for f in gguf_files if "q5_k_m" in f.lower() or "Q5_K_M" in f]
        mmproj_files = [f for f in gguf_files if "mmproj" in f.lower()]
        if not q5_files:
            non_mmproj = [f for f in gguf_files if "mmproj" not in f.lower()]
            if non_mmproj:
                q5_files = sorted(non_mmproj, key=lambda f: os.path.getsize(f), reverse=True)
            else:
                q5_files = gguf_files
        target = f"{{WORK}}/model-q5_k_m.gguf"
        if q5_files[0] != target:
            shutil.copy2(q5_files[0], target)
            print(f"[STEP] Copied {{q5_files[0]}} -> {{target}}")
        if mmproj_files:
            mmproj_target = f"{{WORK}}/model-mmproj.gguf"
            if mmproj_files[0] != mmproj_target:
                shutil.copy2(mmproj_files[0], mmproj_target)
                print(f"[STEP] Copied mmproj: {{mmproj_files[0]}} -> {{mmproj_target}}")
        q5_size = os.path.getsize(target) / (1024**3)
        print(f"[STEP] q5_k_m GGUF created! ({{q5_size:.1f}}GB)")
        if q5_size < MIN_Q5_GB:
            print(f"[ERROR] GGUF too small! Expected at least {{MIN_Q5_GB}}GB but got {{q5_size:.1f}}GB")
            sys.exit(1)
        print("[STEP] Direct GGUF save complete!")
    except Exception as e:
        print(f"[ERROR] Direct GGUF save failed: {{e}}")
        sys.exit(1)
else:
    print("[STEP] Saving merged model...")
    if not check_disk(40, "merge model"):
        sys.exit(1)
    try:
        if IS_MOE_MODEL:
            print("[STEP] MoE model detected. Using PEFT merge path instead of Unsloth merge.")
            try:
                del model
                del trainer
                torch.cuda.empty_cache()
            except Exception:
                pass
            merge_lora_with_peft(FINAL_LORA_DIR, os.path.join(WORK, "gguf_model"))
        else:
            model.save_pretrained_merged(os.path.join(WORK, "gguf_model"), tokenizer)
        print("[STEP] Merged model saved!")
    except Exception as e:
        print(f"[ERROR] Failed to save merged model: {{e}}")
        print("[STEP] Merge failed, but final LoRA adapter should still be saved at:")
        print(FINAL_LORA_DIR)
        sys.exit(1)

    # Fix tokenizer compatibility (Gemma 4 extra_special_tokens list->dict)
    fix_tokenizer_config(os.path.join(WORK, "gguf_model"))

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
        subprocess.run(["pip", "uninstall", "torchvision", "-y"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        quantize_bin = ensure_llama_cpp()
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
        # quantize_bin was prepared by ensure_llama_cpp() during bf16 conversion.
        if not os.path.exists(quantize_bin):
            print(f"[ERROR] llama.cpp quantizer not found at {{quantize_bin}}")
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
    print("[STEP] Uploading to HuggingFace with Xet disabled and private repo setting...")
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        api.create_repo(repo_id=HF_REPO, repo_type="model", exist_ok=True, private=True)
        try:
            api.update_repo_settings(repo_id=HF_REPO, repo_type="model", private=True)
        except Exception:
            pass
        api.upload_file(
            path_or_fileobj=f"{{WORK}}/model-q5_k_m.gguf",
            path_in_repo="model-q5_k_m.gguf",
            repo_id=HF_REPO,
            repo_type="model",
        )
        mmproj_path = f"{{WORK}}/model-mmproj.gguf"
        if os.path.exists(mmproj_path):
            print("[STEP] Uploading mmproj (multimodal projector)...")
            api.upload_file(
                path_or_fileobj=mmproj_path,
                path_in_repo="model-mmproj.gguf",
                repo_id=HF_REPO,
                repo_type="model",
            )
            print("[STEP] mmproj uploaded!")
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


def run_ssh_command(ssh, command, timeout=7200, fail_on_error=False):
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
        if fail_on_error:
            raise RuntimeError(f"SSH command failed: {command}")
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

# Disable Xet for large file upload reliability.
os.environ["HF_HUB_DISABLE_XET"] = "1"

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
api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True, private=True)

print("[HF] Keeping repo private if possible...")
try:
    api.update_repo_settings(repo_id=REPO_ID, repo_type="model", private=True)
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
    elif source == "ready (already cleaned)":
        pairs = data
    else:
        print("    ❌ Unknown format.")
        input("\n    Press Enter to go back...")
        return

    print(f"    ✅ Extracted {len(pairs)} training pairs.")

    hidden_stats = summarize_hidden_internal_blocks(pairs)
    if hidden_stats["pairs_with_hidden_internal_blocks"] > 0:
        print("    🧠 Hidden Claude internal blocks detected and stripped before training:")
        print(f"       - pairs with hidden/internal blocks: {hidden_stats['pairs_with_hidden_internal_blocks']}")
        print(f"       - assistant messages with hidden thinking: {hidden_stats['assistant_messages_with_thinking']}")
        print(f"       - assistant messages with tool blocks: {hidden_stats['assistant_messages_with_tools']}")
        if hidden_stats["user_messages_with_hidden_internal_blocks"] > 0:
            print(f"       - user messages with hidden/internal blocks: {hidden_stats['user_messages_with_hidden_internal_blocks']}")
        if hidden_stats["unknown_structured_blocks"] > 0:
            print(f"       - unknown structured blocks skipped: {hidden_stats['unknown_structured_blocks']}")
    else:
        print("    🧠 No hidden Claude internal blocks detected.")

    print("    🧹 Cleaning visible thinking traces and rejecting contaminated samples...")
    pairs, thinking_removed, rejected = clean_training_data(pairs)

    if thinking_removed > 0:
        print(f"    ✅ Cleaned visible thinking-like prefixes from {thinking_removed} responses.")
    else:
        print("    ✅ No removable visible thinking prefix found.")

    if rejected:
        report_path = save_rejected_report(rejected, file_path)
        print(f"    ⚠️ Rejected {len(rejected)} contaminated samples.")
        if report_path:
            print(f"    📝 Rejected sample report: {report_path}")
    else:
        print("    ✅ No contaminated samples rejected.")

    try:
        assert_no_thinking_in_dataset(pairs)
    except RuntimeError as e:
        print(f"    ❌ {e}")
        input("\n    Press Enter to go back...")
        return

    if len(pairs) == 0:
        print("    ❌ No clean training pairs found after contamination filtering.")
        input("\n    Press Enter to go back...")
        return

    print(f"    ✅ Clean training pairs ready: {len(pairs)}")

    input("\n    Press Enter to continue...")
    model_info = select_model()
    if not model_info:
        input("\n    ❌ Invalid model. Press Enter to go back...")
        return

    # Epoch selection
    clear()
    banner()
    print("    🔄 Select training epochs:\n")
    print("    Epochs = how many times the model trains on your data.")
    print("    More epochs = deeper learning, but risk of overfitting.\n")
    print("    1. 1 epoch  (recommended for most cases)")
    print("    2. 2 epochs (for small datasets < 500 pairs)")
    print("    3. 3 epochs (for very small datasets < 200 pairs)")
    print("    4. Custom")
    print()
    epoch_choice = input("    Select (default 1): ").strip()
    if epoch_choice == "2":
        num_epochs = 2
    elif epoch_choice == "3":
        num_epochs = 3
    elif epoch_choice == "4":
        try:
            num_epochs = int(input("    Enter number of epochs: ").strip())
            if num_epochs < 1:
                num_epochs = 1
            elif num_epochs > 10:
                print("    ⚠️ More than 10 epochs is not recommended. Setting to 10.")
                num_epochs = 10
        except ValueError:
            num_epochs = 1
    else:
        num_epochs = 1
    model_info["_epochs"] = num_epochs

    clear()
    banner()
    print("    📋 Summary:\n")
    print(f"    Data:   {source.upper()}")
    print(f"    Pairs:  {len(pairs)}")
    print(f"    Model:  {model_info['name']}")
    print(f"    GPU:    {model_info['gpu_label']}")
    print(f"    Cost:   {model_info['cost']}")
    print(f"    Epochs: {model_info.get('_epochs', 1)}")
    if model_info.get("gpu_count", 1) > 1:
        print(f"    GPUs:   {model_info.get('gpu_count')}x (multi-GPU)")
    if model_info.get("is_moe"):
        print("    LoRA:   16-bit (MoE model detected)")
    else:
        print("    LoRA:   QLoRA 4-bit")
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
    try:
        import runpod
    except Exception as e:
        print("    ❌ Local dependency missing or broken: runpod")
        print(f"    Error: {e}")
        print("\n    Fix on Windows CMD:")
        print("    py -m pip install --upgrade runpod paramiko huggingface_hub")
        input("\n    Press Enter to go back...")
        return

    clear()
    banner()
    print("    🔥 Starting fine-tuning...\n")
    runpod.api_key = config["runpod_api_key"]
    print(f"    Model: {model_info['name']}")
    print(f"    GPU:   {model_info['gpu_label']}")
    print(f"    Cost:  {model_info['cost']}")
    if model_info.get("gpu_count", 1) > 1:
        print(f"    Mode:  Multi-GPU ({model_info.get('gpu_count')}x)")
    if model_info.get("is_moe"):
        print(f"    LoRA:  16-bit LoRA (MoE)")
    else:
        print(f"    LoRA:  QLoRA 4-bit")
    print()

    pod_id = None
    ssh = None
    upload_success = False

    # [1/6] Create Pod
    print("    [1/6] Creating RunPod instance...")
    gpu_count = model_info.get("gpu_count", 1)
    disk_size = 500 if gpu_count > 1 else 250
    try:
        pod = runpod.create_pod(
            name="respark-finetune",
            image_name="runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04",
            gpu_type_id=model_info["gpu"],
            gpu_count=gpu_count,
            cloud_type="SECURE",
            volume_in_gb=disk_size,
            container_disk_in_gb=disk_size,
            ports="22/tcp",
        )
        pod_id = pod["id"]
        print(f"    ✅ Pod created: {pod_id} ({gpu_count}x GPU)")
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
    if model_info.get("gpu_count", 1) > 1:
        print(f"    (Multi-GPU: {model_info.get('gpu_count')}x — may take 5-10+ hours for large models)\n")
    else:
        print("    (This may take 3-5 hours for 31B)\n")
    try:
        print("    Installing system packages...")
        run_ssh_command(
            ssh,
            "bash -lc 'set -o pipefail; apt-get update && apt-get install -y cmake libcurl4-openssl-dev libssl-dev git build-essential 2>&1 | tail -50'",
            timeout=1800,
            fail_on_error=True,
        )
        print("    ✅ System packages installed!")

        print("    Installing Python packages...")
        install_commands = [
            "pip install --upgrade pip",
            # RunPod/PyTorch images can carry torchaudio/torchtext builds that no longer match
            # the torch version installed by Unsloth. Remove them before and after package upgrades.
            "pip uninstall -y torchaudio torchtext",
            "pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo",
            "pip install xformers trl peft accelerate bitsandbytes datasets huggingface_hub hf_transfer",
            "pip uninstall -y torchaudio torchtext",
            "pip install torchvision",
            "pip uninstall -y torchaudio torchtext",
        ]

        if "gemma-4-12b" in model_info.get("hf_id", "").lower():
            install_commands.append(
                "pip install --upgrade --force-reinstall --no-cache-dir git+https://github.com/huggingface/transformers.git"
            )
        for cmd in install_commands:
            run_ssh_command(
                ssh,
                f"bash -lc 'set -o pipefail; {cmd} 2>&1 | tail -80'",
                timeout=1800,
                fail_on_error=True,
            )

        print("    Checking installed package versions...")
        version_check_cmd = """python - <<'PY'
import importlib.metadata as m

for pkg in ["unsloth", "unsloth_zoo", "transformers", "torch", "trl", "peft", "bitsandbytes"]:
    try:
        print(pkg + ': ' + m.version(pkg))
    except Exception as e:
        print(pkg + ': unavailable (' + str(e) + ')')
PY"""
        run_ssh_command(
            ssh,
            f"bash -lc {shlex.quote(version_check_cmd)}",
            timeout=120,
            fail_on_error=True,
        )
        print("    ✅ Python packages installed!")

        print("    Skipping llama.cpp pre-install. It will be prepared after training, only when GGUF conversion starts.")

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

        gpu_count = model_info.get("gpu_count", 1)
        if gpu_count > 1:
            print(f"\n    🔧 Setting up multi-GPU ({gpu_count}x)...")
            accel_config = f"""compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_machines: 1
num_processes: {gpu_count}
mixed_precision: bf16
use_cpu: false"""
            run_ssh_command(ssh, f"mkdir -p ~/.cache/huggingface/accelerate && echo '{accel_config}' > ~/.cache/huggingface/accelerate/default_config.yaml", timeout=30)
            train_cmd = f"nohup accelerate launch --num_processes={gpu_count} --mixed_precision=bf16 {WORK_DIR}/train.py > {WORK_DIR}/train.log 2>&1 &"
            print(f"\n    🔥 Training started ({gpu_count}x GPU, nohup mode)! Monitoring log...\n")
        else:
            train_cmd = f"nohup python -u {WORK_DIR}/train.py > {WORK_DIR}/train.log 2>&1 &"
            print("\n    🔥 Training started (nohup mode)! Monitoring log...\n")
        run_ssh_command(ssh, train_cmd, timeout=60)
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
