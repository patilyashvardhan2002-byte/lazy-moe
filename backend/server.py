"""
LazyMoE API Server v0.3
Universal model support up to 200B+ parameters.
Auto-detects model architecture and configures all components.

Endpoints:
  GET  /health   — server + model status
  POST /infer    — inference via SSE token stream
  GET  /cache    — expert cache state
  GET  /kv       — TurboQuant KV stats
  GET  /model    — detected model architecture
  GET  /stats    — full system stats
  GET  /system   — hardware info + model compatibility
  POST /reset    — clear all caches
"""

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from expert_cache import ExpertCache
from query_analyzer import QueryAnalyzer
from turboquant import KVCacheManager
from llama_bridge import LlamaBridge, InferenceConfig
from model_detector import ModelDetector, ModelConfig
from system_detector import SystemDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("lazy-moe.server")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get(
    "LAZY_MOE_MODEL",
    "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
)
SHARD_DIR  = os.environ.get("LAZY_MOE_SHARDS", "./models/shards")
RAM_BUDGET = float(os.environ.get("LAZY_MOE_RAM_GB", "8.0"))
N_THREADS  = int(os.environ.get("LAZY_MOE_THREADS", "4"))

# ── Global state ──────────────────────────────────────────────────────────────
model_config: ModelConfig = None
expert_cache: ExpertCache = None
query_analyzer: QueryAnalyzer = None
kv_manager: KVCacheManager = None
llama_bridge: LlamaBridge = None
_system_cache = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_config, expert_cache, query_analyzer, kv_manager, llama_bridge

    logger.info("=== LazyMoE Starting ===")

    # 1. Detect model architecture
    detector = ModelDetector(ram_budget_gb=RAM_BUDGET)
    model_config = detector.detect(MODEL_PATH)
    logger.info(f"Model: {model_config.summary}")

    # 2. Expert cache (LRU, mmap-based)
    expert_cache = ExpertCache(
        capacity=model_config.expert_cache_slots,
        shard_dir=SHARD_DIR,
        ram_budget_gb=RAM_BUDGET - 3.5,
    )

    # 3. Query domain classifier
    query_analyzer = QueryAnalyzer(mode="keyword")

    # 4. TurboQuant KV cache manager
    kv_manager = KVCacheManager(
        num_layers=model_config.num_layers,
        dim=max(model_config.kv_dim, 64),
        bits=model_config.kv_cache_bits,
        ram_budget_gb=RAM_BUDGET * 0.4,
    )

    # 5. llama.cpp bridge (starts llama-server in background)
    threads = int(os.environ.get("LAZY_MOE_THREADS", str(model_config.recommended_threads)))
    inference_config = InferenceConfig(
        model_path=MODEL_PATH,
        n_threads=threads,
        n_ctx=model_config.recommended_ctx,
        n_predict=model_config.recommended_n_predict,
        mmap=True,
        mlock=False,
    )

    def on_expert_activate(expert_ids):
        for eid in expert_ids:
            expert_cache.get(eid)

    llama_bridge = LlamaBridge(
        config=inference_config,
        on_expert_activate=on_expert_activate
    )

    logger.info(
        f"Ready — {model_config.name} | "
        f"layers={model_config.num_layers} | "
        f"moe={model_config.is_moe} | "
        f"experts={model_config.num_experts} | "
        f"cache_slots={model_config.expert_cache_slots} | "
        f"kv_bits={model_config.kv_cache_bits} | "
        f"ctx={model_config.recommended_ctx} | "
        f"ram_budget={RAM_BUDGET}GB"
    )
    yield
    if llama_bridge:
        llama_bridge.stop()
    logger.info("LazyMoE shutdown complete")


app = FastAPI(title="LazyMoE API", version="0.3.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request models ────────────────────────────────────────────────────────────
class InferRequest(BaseModel):
    query: str
    system_prompt: str = "You are a helpful assistant. Be concise."
    max_tokens: int = 400
    temperature: float = 0.7


class ResetRequest(BaseModel):
    clear_kv: bool = True
    clear_experts: bool = True


# ── SSE helper ────────────────────────────────────────────────────────────────
def sse(event_type: str, data: dict) -> str:
    return f"data: {json.dumps({'type': event_type, **data})}\n\n"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_available": llama_bridge.is_available(),
        "model_path": MODEL_PATH,
        "mock_mode": not llama_bridge.is_available(),
        "model": {
            "name":        model_config.name,
            "family":      model_config.family,
            "params_b":    model_config.params_b,
            "size_gb":     round(model_config.model_size_gb, 2),
            "layers":      model_config.num_layers,
            "is_moe":      model_config.is_moe,
            "num_experts": model_config.num_experts,
            "num_kv_heads":model_config.num_kv_heads,
            "quant":       model_config.quant,
            "ctx":         model_config.recommended_ctx,
            "kv_bits":     model_config.kv_cache_bits,
            "expert_cache_slots": model_config.expert_cache_slots,
        },
        "ram_budget_gb": RAM_BUDGET,
    }


@app.post("/infer")
async def infer(req: InferRequest):
    async def event_stream():
        t0 = time.time()
        try:
            # Phase 1: Analyze query
            yield sse("phase", {"phase": "analyzing"})
            analysis = query_analyzer.analyze(req.query)
            yield sse("analysis", {
                "domain":          analysis.domain,
                "confidence":      round(analysis.confidence, 3),
                "active_experts":  analysis.active_experts,
                "fallback_experts":analysis.fallback_experts,
                "tokens_estimate": analysis.tokens_estimate,
                "analysis_ms":     analysis.analysis_ms,
            })
            await asyncio.sleep(0.05)

            # Phase 2: Prefetch experts
            yield sse("phase", {"phase": "prefetching"})
            for expert_id in analysis.active_experts:
                load_start = time.perf_counter()
                was_cached = expert_cache._cache.__contains__(expert_id)
                await asyncio.to_thread(expert_cache.get, expert_id)
                load_ms = (time.perf_counter() - load_start) * 1000
                yield sse("expert", {
                    "expert_id":     expert_id,
                    "hit":           was_cached,
                    "load_ms":       round(load_ms, 1),
                    "cache_snapshot":expert_cache.snapshot(),
                    "cache_stats":   expert_cache.stats,
                })
                await asyncio.sleep(0.01)

            # Phase 3: Inference
            yield sse("phase", {"phase": "inferring"})
            llama_bridge.config.n_predict = req.max_tokens
            llama_bridge.config.temp = req.temperature

            token_events = await asyncio.to_thread(
                lambda: list(llama_bridge.stream(req.query, req.system_prompt))
            )

            token_count = 0
            for i, event in enumerate(token_events):
                token_count += 1
                yield sse("token", {
                    "token":          event.token,
                    "token_id":       i,
                    "active_experts": event.active_experts,
                })
                if i % 10 == 0:
                    kv = kv_manager.summary()
                    yield sse("kv_update", {
                        "tokens":            token_count,
                        "ram_used_gb":       kv["ram_used_gb"],
                        "raw_fp16_gb":       kv["raw_fp16_gb"],
                        "compression_ratio": kv["compression_ratio"],
                    })
                await asyncio.sleep(0)

            elapsed = time.time() - t0
            tok_sec = token_count / elapsed if elapsed > 0 else 0
            yield sse("done", {
                "tokens":          token_count,
                "elapsed_sec":     round(elapsed, 2),
                "tokens_per_sec":  round(tok_sec, 1),
                "cache_stats":     expert_cache.stats,
                "kv_stats":        kv_manager.summary(),
                "mock_mode":       not llama_bridge.is_available(),
            })

        except asyncio.CancelledError:
            llama_bridge.stop()
            yield sse("error", {"message": "Inference cancelled"})
        except Exception as e:
            logger.exception(f"Inference error: {e}")
            yield sse("error", {"message": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/cache")
async def get_cache():
    return {
        "snapshot": expert_cache.snapshot(),
        "stats":    expert_cache.stats,
        "ram_used_gb": expert_cache.ram_used_gb,
    }


@app.get("/kv")
async def get_kv():
    return kv_manager.summary()


@app.get("/model")
async def get_model():
    return {
        "name":             model_config.name,
        "family":           model_config.family,
        "params_b":         model_config.params_b,
        "size_gb":          round(model_config.model_size_gb, 2),
        "layers":           model_config.num_layers,
        "num_heads":        model_config.num_heads,
        "num_kv_heads":     model_config.num_kv_heads,
        "head_dim":         model_config.head_dim,
        "hidden_dim":       model_config.hidden_dim,
        "is_moe":           model_config.is_moe,
        "num_experts":      model_config.num_experts,
        "experts_per_token":model_config.experts_per_token,
        "quant":            model_config.quant,
        "context_length":   model_config.context_length,
        "recommended_ctx":  model_config.recommended_ctx,
        "kv_bits":          model_config.kv_cache_bits,
        "expert_cache_slots":model_config.expert_cache_slots,
        "kv_per_token_fp16_gb": round(model_config.kv_ram_fp16_per_token_gb, 6),
        "kv_per_token_tq_gb":   round(model_config.kv_ram_tq_per_token_gb, 6),
    }


@app.get("/stats")
async def get_stats():
    kv = kv_manager.summary()
    return {
        "model": {
            "name":     model_config.name,
            "params_b": model_config.params_b,
        },
        "expert_cache": expert_cache.stats,
        "kv_cache":     kv,
        "ram": {
            "budget_gb":    RAM_BUDGET,
            "attention_gb": 3.2,
            "experts_gb":   expert_cache.ram_used_gb,
            "kv_gb":        kv["ram_used_gb"],
            "total_gb":     round(3.2 + expert_cache.ram_used_gb + kv["ram_used_gb"], 3),
        },
    }


@app.get("/system")
async def get_system():
    global _system_cache
    if _system_cache is None:
        detector = SystemDetector()
        info = detector.detect()
        compat = detector.get_model_compatibility(info)
        _system_cache = {
            "os": {
                "name":    info.os_name,
                "version": info.os_version[:60],
                "arch":    info.architecture,
            },
            "cpu": {
                "name":           info.cpu_name[:80],
                "cores_physical": info.cpu_cores_physical,
                "cores_logical":  info.cpu_cores_logical,
                "freq_mhz":       round(info.cpu_freq_mhz),
                "vendor":         info.cpu_vendor,
            },
            "ram": {
                "total_gb":     round(info.ram_total_gb, 1),
                "available_gb": round(info.ram_available_gb, 1),
                "used_pct":     round(info.ram_used_pct, 1),
                "effective_gb": round(info.effective_ram_gb, 1),
            },
            "gpu": {
                "devices": [
                    {
                        "name":       g.name,
                        "vram_gb":    g.vram_gb,
                        "vendor":     g.vendor,
                        "integrated": g.is_integrated,
                    }
                    for g in info.gpus
                ],
                "has_discrete":   info.has_discrete_gpu,
                "total_vram_gb":  round(info.total_vram_gb, 1),
            },
            "disk": {
                "total_gb": round(info.disk_total_gb, 1),
                "free_gb":  round(info.disk_free_gb, 1),
            },
            "apple_silicon":     info.is_apple_silicon,
            "unified_memory_gb": round(info.unified_memory_gb, 1),
            "model_compatibility": compat,
        }
    return _system_cache


@app.post("/reset")
async def reset(req: ResetRequest):
    global kv_manager
    if req.clear_experts:
        expert_cache.clear()
    if req.clear_kv:
        kv_manager = KVCacheManager(
            num_layers=model_config.num_layers,
            dim=max(model_config.kv_dim, 64),
            bits=model_config.kv_cache_bits,
        )
    return {"status": "reset", "cleared_experts": req.clear_experts, "cleared_kv": req.clear_kv}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
