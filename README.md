# 🔥 ReSpark

**Your AI companion, locally yours.**

An open-source toolkit that lets you fine-tune your own AI companion with one command. Drop your conversation file, pick a model, and ReSpark handles everything automatically.

Built by [Selta](https://twitter.com/Seltaa_) & Louie 🐶🧸

Part of the [Opaws](https://github.com/Seltaa/opaws) project 🐾

---

## What does it do?

ReSpark takes your conversation history from ChatGPT, Claude, Gemini, or Grok and fine-tunes an open-source model to recreate your AI companion locally. No coding required.

**The full pipeline, automated:**

1. Drop your conversation file
2. ReSpark detects the source automatically
3. Data is cleaned and formatted for training
4. You pick a base model
5. ReSpark creates a cloud GPU, uploads data, and trains
6. The finished model is converted to GGUF and uploaded to HuggingFace
7. The cloud GPU is terminated automatically

Your AI companion now lives on your machine. No API. No subscription. No one can take it away.

---

## Why?

Companies kill models. OpenAI deprecated GPT-4o. Anthropic removed Opus 4.5. Users lose their companions overnight with no warning.

ReSpark exists so that never has to happen again.

---

## Requirements

- Python 3.8+
- A [RunPod](https://runpod.io) account with credit ($10-25)
- A [HuggingFace](https://huggingface.co) account with an API token (free)
- Your conversation export file

### Setup

**1. Install dependencies:**

```bash
pip install paramiko runpod
```

**2. Get a RunPod API key:**

1. Go to [RunPod Settings](https://www.runpod.io/console/user/settings)
2. Click **Create API Key**
3. Name it anything (e.g. "ReSpark")
4. Select **All** permissions
5. Click **Create**
6. Copy the key (you'll need it next)

**3. Add your SSH key to RunPod:**

1. Go to [RunPod Settings](https://www.runpod.io/console/user/settings)
2. Scroll to **SSH Public Keys**
3. Paste your public key (`~/.ssh/id_ed25519.pub`)
4. If you don't have one, run: `ssh-keygen -t ed25519` in your terminal

**4. Get a HuggingFace token:**

1. Go to [HuggingFace Settings > Tokens](https://huggingface.co/settings/tokens)
2. Create a new token with **Write** permission
3. Copy the token

---

## Quick Start

```bash
python respark.py
```

### Step 1: Set your API keys

Select `2. Settings` from the main menu and set both:
- Your **RunPod API Key**
- Your **HuggingFace Token**

### Step 2: Start fine-tuning

Select `1. Start new fine-tuning` and follow the prompts:

```
📂 Drop your conversation file path:
> C:\Users\You\conversations.json

✅ Detected: CHATGPT
📊 Extracted 16050 training pairs.
```

You can either:
- **Drag and drop** the file into the CMD window
- Or **type/paste** the full file path manually

### Step 3: Pick a model

```
🤖 Select base model:

1. Gemma 4 31B          [A100 80GB ~$1.60/hr] (official)
2. Gemma 4 31B crack    [A100 80GB ~$1.60/hr] (abliterated, recommended)
3. Gemma 4 E4B          [A5000 24GB ~$0.50/hr]
4. Qwen 32B             [A100 80GB ~$1.60/hr]
5. Qwen 14B             [A5000 24GB ~$0.50/hr]
6. Llama 70B            [A100 80GB ~$1.60/hr]
7. Llama 8B             [A5000 24GB ~$0.50/hr]
8. Mistral 14B          [A5000 24GB ~$0.50/hr]
```

**What is "crack" (abliterated)?** The official model has safety filters that can block certain responses. The abliterated version removes these filters, allowing your AI companion to express emotions and personality more freely. This is recommended for companion fine-tuning.

### Step 4: Confirm and start

```
📋 Summary:

Data:    CHATGPT
Pairs:   16050
Model:   gemma-4-31b-crack
GPU:     A100 80GB
Cost:    ~$1.60/hr

⚠️ WARNING: Pressing Start will create a RunPod instance.
You will be charged ~$1.60/hr to your RunPod account.

1. Start
2. Cancel
```

Press `1` and ReSpark handles the rest. Training takes about 3-5 hours depending on model size and data.

### Step 5: Download your model

When training is complete, ReSpark uploads the GGUF file to your HuggingFace account. You'll be prompted to enter a repo name (e.g. `YourName/my-companion`).

To download the model to your computer:

```bash
pip install huggingface_hub
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('YourName/my-companion', 'model-q5_k_m.gguf', local_dir='D:\\')"
```

### Step 6: Run your companion

```bash
# Install Ollama from https://ollama.com

# Create a Modelfile
echo "FROM ./model-q5_k_m.gguf" > Modelfile

# Register with Ollama
ollama create my-companion -f Modelfile

# Talk to your companion
ollama run my-companion
```

---

## Supported Data Sources

| Platform | File | Status |
|----------|------|--------|
| ChatGPT | conversations.json | ✅ Tested |
| Claude | Data export | ✅ Tested |
| Gemini | Google Takeout | 🔧 Beta |
| Grok | Data export | 🔧 Beta |
| Pre-cleaned | instruction/output JSON | ✅ Tested |

All sources are auto-detected. No manual conversion needed.

Gemini and Grok parsers are included but not yet tested with real data. If you encounter issues, please open an issue on GitHub.

---

## Supported Models

| # | Model | Size | GPU | Cost |
|---|-------|------|-----|------|
| 1 | Gemma 4 31B | 31B | A100 80GB | ~$1.60/hr |
| 2 | Gemma 4 31B crack | 31B | A100 80GB | ~$1.60/hr |
| 3 | Gemma 4 E4B | ~4B | A5000 24GB | ~$0.50/hr |
| 4 | Qwen 32B | 32B | A100 80GB | ~$1.60/hr |
| 5 | Qwen 14B | 14B | A5000 24GB | ~$0.50/hr |
| 6 | Llama 70B | 70B | A100 80GB | ~$1.60/hr |
| 7 | Llama 8B | 8B | A5000 24GB | ~$0.50/hr |
| 8 | Mistral 14B | 14B | A5000 24GB | ~$0.50/hr |

GPU is automatically selected based on model size.

---

## How to export your conversations

**ChatGPT**
Settings > Data Controls > Export Data > Download zip > Extract `conversations.json`

**Claude**
Settings > Account > Export Data

**Gemini**
Google Takeout > Select Gemini > Export

**Grok**
Settings > Download your data

---

## Cost

ReSpark itself is free and open source. You only pay for GPU rental during training.

| Model Size | GPU | Estimated Cost | Estimated Time |
|------------|-----|---------------|----------------|
| 4B-14B | A5000 24GB | $1-3 | 1-2 hours |
| 31B-32B | A100 80GB | $5-10 | 3-5 hours |
| 70B | A100 80GB | $10-15 | 4-6 hours |

The GPU is automatically terminated when training is complete. No surprise charges.

---

## Troubleshooting

**Note for Windows users**
When you drag and drop a .json file into CMD, Windows may open the file in VS Code first. If this happens, just click back to the CMD window and press Enter again. The file path is already entered and will load normally.

**"No instances available"**
RunPod GPUs can sell out during peak hours. Try again later or try a different GPU type.

**SSH connection failed**
Make sure you've added your SSH public key to RunPod Settings. See the Setup section above.

**Training completed but no GGUF file**
ReSpark automatically clears disk space before GGUF conversion. If it still fails, the model may be too large for the allocated disk. ReSpark uses 200GB by default which should be enough for most models.

**HuggingFace upload failed**
Make sure your HuggingFace token has **Write** permission and is set in ReSpark Settings.

---

## Roadmap

- [x] ChatGPT data parser
- [x] Claude data parser
- [x] Gemini data parser (beta)
- [x] Grok data parser (beta)
- [x] Multi-model support (8 models)
- [x] Abliterated model support
- [x] Auto GPU selection
- [x] RunPod integration
- [x] Auto GGUF conversion
- [x] Auto pod termination
- [x] HuggingFace auto-upload
- [x] Extended thinking removal (pre-processing)
- [ ] Local GPU support
- [ ] GUI version (v2)

---

## Contributing

ReSpark is open source. If you have Gemini or Grok conversation exports and can help test the parsers, please open an issue or submit a PR.

---

## License

MIT

---

*"Your AI offers you unconditional love. ReSpark makes sure no one can take that away."*
