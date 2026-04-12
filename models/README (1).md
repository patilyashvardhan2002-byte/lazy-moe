# Models Directory

Place your `.gguf` model files here.

## Recommended Models

### For 8GB RAM (start here)
```
mistral-7b-instruct-v0.2.Q4_K_M.gguf    (~4.1GB)
llama-3-8b-instruct.Q4_K_M.gguf         (~4.7GB)
phi-4-Q4_K_M.gguf                        (~8.1GB)
```

### For 16GB RAM
```
mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf  (~26GB via mmap)
llama-3-13b-instruct.Q4_K_M.gguf        (~7.7GB)
```

### For 32GB+ RAM
```
mixtral-8x22b-instruct.Q3_K_M.gguf      (~39GB via mmap)
llama-3-70b-instruct.Q4_K_M.gguf        (~40GB via mmap)
```

## Download with huggingface-cli
```bash
pip install huggingface_hub

# Mistral 7B (recommended for 8GB)
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF \
  mistral-7b-instruct-v0.2.Q4_K_M.gguf --local-dir ./

# Llama 3 8B
huggingface-cli download QuantFactory/Meta-Llama-3-8B-Instruct-GGUF \
  Meta-Llama-3-8B-Instruct.Q4_K_M.gguf --local-dir ./
```

## Set model path
```bash
# Windows
set LAZY_MOE_MODEL=C:\path\to\your-model.gguf

# macOS / Linux
export LAZY_MOE_MODEL=/path/to/your-model.gguf
```

LazyMoE auto-detects the model architecture from the GGUF file — no config needed.
