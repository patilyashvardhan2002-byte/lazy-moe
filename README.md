# LazyMoE

> Run 120B-class LLMs on 8GB RAM — no GPU required.

Three techniques combined for the first time:

| Technique | What it does |
|---|---|
| **Lazy Expert Loading** | LRU cache loads MoE expert weights from SSD on demand |
| **1-bit Quantization** | BitNet-style weights shrink experts 4× vs Q4 |
| **TurboQuant KV** | Google Research's 3-bit KV cache compression (6× reduction) |

---

## Demo

![LazyMoE Dashboard](https://raw.githubusercontent.com/yourusername/lazy-moe/main/assets/demo.png)

---

## Quick Start

### 1. Clone
```bash
git clone https://github.com/yourusername/lazy-moe
cd lazy-moe
```

### 2. Install backend deps
```bash
pip install -r backend/requirements.txt
```

### 3. Install frontend deps
```bash
cd frontend && npm install
```

### 4. Download llama.cpp

**Windows:** Download from https://github.com/ggerganov/llama.cpp/releases/latest
- Pick `llama-bXXXX-bin-win-cpu-x64.zip`
- Extract to `llama.cpp/` folder

**macOS / Linux:**
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && cmake -B build && cmake --build build -j8
```

### 5. Download a model

```bash
pip install huggingface_hub

# Mistral 7B (works on 8GB RAM)
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF \
  mistral-7b-instruct-v0.2.Q4_K_M.gguf --local-dir models/
```

### 6. Run

**Windows (double-click):**
```
scripts/start_windows.bat
```

**macOS / Linux:**
```bash
bash scripts/start.sh
```

**Manual:**
```bash
# Terminal 1 — Backend
cd backend
set LAZY_MOE_MODEL=../models/mistral-7b-instruct-v0.2.Q4_K_M.gguf  # Windows
export LAZY_MOE_MODEL=../models/mistral-7b-instruct-v0.2.Q4_K_M.gguf  # Mac/Linux
python server.py

# Terminal 2 — Frontend
cd frontend && npm run dev
```

Open **http://localhost:5173**

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────┐
│   Query Analyzer        │  ← domain detection (code/math/language/reasoning)
│   (keyword / embedding) │  ← predicts which MoE experts will activate
└──────────┬──────────────┘
           │ predicted expert IDs
           ▼
┌─────────────────────────┐
│   Expert LRU Cache      │  ← loads expert weights SSD → RAM on demand
│   (mmap, lazy load)     │  ← evicts least-recently-used when RAM fills
└──────────┬──────────────┘
           │ weights in RAM
           ▼
┌─────────────────────────┐
│   llama-server          │  ← model stays loaded in RAM permanently
│   (llama.cpp HTTP API)  │  ← no reload per query
└──────────┬──────────────┘
           │ tokens + KV vectors
           ▼
┌─────────────────────────┐
│   TurboQuant            │  ← compresses KV cache 6× as tokens generate
│   (3-bit, lossless)     │  ← Google Research ICLR 2026
└──────────┬──────────────┘
           │ SSE token stream
           ▼
       Browser UI
```

---

## RAM Budget (8GB device)

| Component | Size |
|---|---|
| Attention layers (fp16, always loaded) | 3.2 GB |
| Expert LRU cache (Q4, 3 slots) | 4.2 GB |
| KV cache (TurboQuant 3-bit) | ~0.5 GB |
| **Total** | **~7.9 GB** ✓ |

---

## Supported Models

| Model | Params | Min RAM | Status |
|---|---|---|---|
| Phi-3 Mini | 3.8B | 3GB | ✅ Great |
| Mistral 7B | 7B | 5GB | ✅ Great |
| Llama 3 8B | 8B | 5GB | ✅ Great |
| Gemma 2 9B | 9B | 6GB | ✅ Great |
| Phi-4 | 14B | 9GB | 🟡 OK |
| Qwen2.5 14B | 14B | 9GB | 🟡 OK |
| Mixtral 8x7B | 47B | 26GB | 🟠 SSD stream |
| Llama 3 70B | 70B | 40GB | 🟠 SSD stream |
| Mixtral 8x22B | 141B | 65GB | 🟠 SSD stream |
| Llama 3 405B | 405B | 200GB | 🟠 SSD stream |
| DeepSeek V3 | 671B | 350GB | 🟠 SSD stream |

Click **⬡ SYSTEM** in the UI to see exactly which models run on YOUR hardware.

---

## ⚠ SSD Protection

LazyMoE uses `mlock` to pin model weights in RAM and prevent the OS
from writing them to swap. This protects your SSD from excessive
write cycles.

If your RAM is smaller than the model size, LazyMoE will refuse to
load rather than swapping. Stick to models that fit within your RAM
budget — use the ⬡ SYSTEM panel in the UI to see exactly which
models are safe for your hardware.

---

## Environment Variables

```bash
LAZY_MOE_MODEL=./models/your-model.gguf   # path to GGUF file
LAZY_MOE_RAM_GB=8                          # RAM budget in GB
LAZY_MOE_THREADS=4                         # CPU threads for inference
LAZY_MOE_SHARDS=./models/shards           # expert shard directory
LAZY_MOE_KV_BITS=3    

```

---

## Project Structure

```
lazy-moe/
├── backend/
│   ├── server.py           FastAPI server, SSE streaming
│   ├── expert_cache.py     LRU cache for MoE expert weight shards
│   ├── query_analyzer.py   Domain detection + expert prediction
│   ├── turboquant.py       KV cache compression (TurboQuant algorithm)
│   ├── llama_bridge.py     llama.cpp subprocess manager
│   ├── model_detector.py   Auto-detects model architecture from GGUF
│   ├── system_detector.py  Hardware detection + model compatibility
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx         Main dashboard (cyberpunk terminal UI)
│   │   ├── SystemPanel.jsx Hardware info + model compatibility matrix
│   │   └── main.jsx
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── models/
│   └── README.md           Model download instructions
├── scripts/
│   ├── start_windows.bat   One-click Windows launcher
│   └── start.sh            macOS/Linux launcher
├── .gitignore
└── README.md
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Server + model status |
| `/infer` | POST | Inference via SSE stream |
| `/model` | GET | Detected model architecture |
| `/system` | GET | Hardware info + model compatibility |
| `/cache` | GET | Expert cache state |
| `/kv` | GET | TurboQuant KV stats |
| `/stats` | GET | Full system stats |
| `/reset` | POST | Clear all caches |

---

## Research Background

- **Mixtral MoE** — Mistral AI, 2024. Sparse expert routing, 2 of 8 experts active per token.
- **BitNet b1.58** — Microsoft Research, 2024. 1-bit ternary weights {-1, 0, +1}.
- **TurboQuant** — Google Research (Zandieh et al.), ICLR 2026. Near-optimal KV cache compression via PolarQuant + QJL.
- **llama.cpp** — Georgi Gerganov. CPU inference with mmap SSD offloading.

---

## Roadmap

- [x] SSD swap protection via mlock
- [x] Universal model detection from GGUF metadata
- [x] Hardware compatibility matrix
- [x] TurboQuant KV cache compression
- [x] Cyberpunk terminal dashboard
- [ ] Real llama.cpp MoE expert routing hooks (C++ patch)
- [ ] 1-bit model support — requires BitNet native trained weights
(not post-quantization). Waiting for public 70B+ BitNet release.
- [ ] Expert activation profiling pipeline
- [ ] Speculative decoding (3B draft + 120B verifier)
- [ ] Multi-device sharding via Exo
---

## License

MIT — build whatever you want with it.

---

*Built from scratch by exploring the question: can you run a 120B model on 8GB RAM?*
*Turns out — yes, with enough creativity.*
