"""
LazyMoE Model Detector
Reads GGUF metadata to auto-detect model architecture.
Supports any model up to 200B parameters.

Detects:
  - Model family (Mistral, Mixtral, Llama, Qwen, DeepSeek, etc.)
  - Parameter count
  - Layer count (for TurboQuant config)
  - Head dimensions (for KV cache sizing)
  - MoE config (num experts, experts per token)
  - Recommended RAM/cache settings
"""

import os
import struct
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("lazy-moe.detector")


@dataclass
class ModelConfig:
    # Identity
    name: str = "unknown"
    family: str = "unknown"
    params_b: float = 0.0          # billions of parameters
    quant: str = "unknown"         # Q4_K_M, Q8_0, etc.
    path: str = ""

    # Architecture
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    hidden_dim: int = 4096
    context_length: int = 4096

    # MoE config
    is_moe: bool = False
    num_experts: int = 1
    experts_per_token: int = 1

    # Derived settings
    kv_dim: int = 0               # num_kv_heads * head_dim
    model_size_gb: float = 0.0    # actual file size

    # Recommended LazyMoE settings
    recommended_threads: int = 4
    recommended_ctx: int = 2048
    recommended_n_predict: int = 512
    expert_cache_slots: int = 3
    kv_cache_bits: int = 3

    def __post_init__(self):
        if self.kv_dim == 0:
            self.kv_dim = self.num_kv_heads * self.head_dim

    @property
    def summary(self) -> str:
        moe = f" MoE({self.num_experts}x)" if self.is_moe else ""
        return (
            f"{self.name} | {self.params_b:.0f}B{moe} | "
            f"{self.num_layers}L | {self.quant} | "
            f"{self.model_size_gb:.1f}GB"
        )

    @property
    def kv_ram_fp16_per_token_gb(self) -> float:
        """RAM needed per token in KV cache at fp16."""
        return (2 * self.num_layers * self.kv_dim * 2) / 1e9

    @property
    def kv_ram_tq_per_token_gb(self) -> float:
        """RAM needed per token with TurboQuant 3-bit."""
        return self.kv_ram_fp16_per_token_gb / (16 / self.kv_cache_bits)


# Known model profiles (fallback when GGUF metadata unavailable)
KNOWN_MODELS = {
    # Mistral family
    "mistral-7b":     ModelConfig(name="Mistral 7B",     family="mistral",  params_b=7,   num_layers=32,  num_heads=32, num_kv_heads=8,  head_dim=128, hidden_dim=4096,  is_moe=False),
    "mistral-12b":    ModelConfig(name="Mistral 12B",    family="mistral",  params_b=12,  num_layers=40,  num_heads=32, num_kv_heads=8,  head_dim=128, hidden_dim=5120,  is_moe=False),

    # Mixtral MoE family
    "mixtral-8x7b":   ModelConfig(name="Mixtral 8x7B",   family="mixtral",  params_b=47,  num_layers=32,  num_heads=32, num_kv_heads=8,  head_dim=128, hidden_dim=4096,  is_moe=True,  num_experts=8, experts_per_token=2),
    "mixtral-8x22b":  ModelConfig(name="Mixtral 8x22B",  family="mixtral",  params_b=141, num_layers=56,  num_heads=48, num_kv_heads=8,  head_dim=128, hidden_dim=6144,  is_moe=True,  num_experts=8, experts_per_token=2),

    # Llama 3 family
    "llama-3-8b":     ModelConfig(name="Llama 3 8B",     family="llama",    params_b=8,   num_layers=32,  num_heads=32, num_kv_heads=8,  head_dim=128, hidden_dim=4096,  is_moe=False),
    "llama-3-70b":    ModelConfig(name="Llama 3 70B",    family="llama",    params_b=70,  num_layers=80,  num_heads=64, num_kv_heads=8,  head_dim=128, hidden_dim=8192,  is_moe=False),
    "llama-3-405b":   ModelConfig(name="Llama 3 405B",   family="llama",    params_b=405, num_layers=126, num_heads=128,num_kv_heads=16, head_dim=128, hidden_dim=16384, is_moe=False),

    # Llama 3.1/3.2/3.3
    "llama-3.1-8b":   ModelConfig(name="Llama 3.1 8B",   family="llama",    params_b=8,   num_layers=32,  num_heads=32, num_kv_heads=8,  head_dim=128, hidden_dim=4096,  is_moe=False),
    "llama-3.1-70b":  ModelConfig(name="Llama 3.1 70B",  family="llama",    params_b=70,  num_layers=80,  num_heads=64, num_kv_heads=8,  head_dim=128, hidden_dim=8192,  is_moe=False),
    "llama-3.3-70b":  ModelConfig(name="Llama 3.3 70B",  family="llama",    params_b=70,  num_layers=80,  num_heads=64, num_kv_heads=8,  head_dim=128, hidden_dim=8192,  is_moe=False),

    # Qwen family
    "qwen2-7b":       ModelConfig(name="Qwen2 7B",       family="qwen",     params_b=7,   num_layers=28,  num_heads=28, num_kv_heads=4,  head_dim=128, hidden_dim=3584,  is_moe=False),
    "qwen2-72b":      ModelConfig(name="Qwen2 72B",      family="qwen",     params_b=72,  num_layers=80,  num_heads=64, num_kv_heads=8,  head_dim=128, hidden_dim=8192,  is_moe=False),
    "qwen2.5-7b":     ModelConfig(name="Qwen2.5 7B",     family="qwen",     params_b=7,   num_layers=28,  num_heads=28, num_kv_heads=4,  head_dim=128, hidden_dim=3584,  is_moe=False),
    "qwen2.5-72b":    ModelConfig(name="Qwen2.5 72B",    family="qwen",     params_b=72,  num_layers=80,  num_heads=64, num_kv_heads=8,  head_dim=128, hidden_dim=8192,  is_moe=False),

    # DeepSeek family
    "deepseek-7b":    ModelConfig(name="DeepSeek 7B",    family="deepseek", params_b=7,   num_layers=30,  num_heads=32, num_kv_heads=8,  head_dim=128, hidden_dim=4096,  is_moe=False),
    "deepseek-v2":    ModelConfig(name="DeepSeek V2",    family="deepseek", params_b=236, num_layers=60,  num_heads=128,num_kv_heads=128,head_dim=128, hidden_dim=5120,  is_moe=True,  num_experts=160,experts_per_token=6),
    "deepseek-v3":    ModelConfig(name="DeepSeek V3",    family="deepseek", params_b=671, num_layers=61,  num_heads=128,num_kv_heads=128,head_dim=128, hidden_dim=7168,  is_moe=True,  num_experts=256,experts_per_token=8),
    "deepseek-r1":    ModelConfig(name="DeepSeek R1",    family="deepseek", params_b=671, num_layers=61,  num_heads=128,num_kv_heads=128,head_dim=128, hidden_dim=7168,  is_moe=True,  num_experts=256,experts_per_token=8),

    # Phi family
    "phi-3-mini":     ModelConfig(name="Phi-3 Mini",     family="phi",      params_b=3.8, num_layers=32,  num_heads=32, num_kv_heads=32, head_dim=96,  hidden_dim=3072,  is_moe=False),
    "phi-3-medium":   ModelConfig(name="Phi-3 Medium",   family="phi",      params_b=14,  num_layers=40,  num_heads=40, num_kv_heads=10, head_dim=128, hidden_dim=5120,  is_moe=False),
    "phi-4":          ModelConfig(name="Phi-4",          family="phi",      params_b=14,  num_layers=40,  num_heads=40, num_kv_heads=10, head_dim=128, hidden_dim=5120,  is_moe=False),

    # Gemma family
    "gemma-2b":       ModelConfig(name="Gemma 2B",       family="gemma",    params_b=2,   num_layers=18,  num_heads=8,  num_kv_heads=1,  head_dim=256, hidden_dim=2048,  is_moe=False),
    "gemma-7b":       ModelConfig(name="Gemma 7B",       family="gemma",    params_b=7,   num_layers=28,  num_heads=16, num_kv_heads=16, head_dim=256, hidden_dim=3072,  is_moe=False),
    "gemma-2-27b":    ModelConfig(name="Gemma 2 27B",    family="gemma",    params_b=27,  num_layers=46,  num_heads=32, num_kv_heads=16, head_dim=128, hidden_dim=4608,  is_moe=False),

    # Command R
    "command-r":      ModelConfig(name="Command R",      family="cohere",   params_b=35,  num_layers=40,  num_heads=32, num_kv_heads=8,  head_dim=128, hidden_dim=4096,  is_moe=False),
    "command-r-plus": ModelConfig(name="Command R+",     family="cohere",   params_b=104, num_layers=64,  num_heads=64, num_kv_heads=16, head_dim=128, hidden_dim=8192,  is_moe=False),

    # Falcon
    "falcon-180b":    ModelConfig(name="Falcon 180B",    family="falcon",   params_b=180, num_layers=80,  num_heads=232,num_kv_heads=8,  head_dim=64,  hidden_dim=14848, is_moe=False),
}


class ModelDetector:
    """
    Detects model architecture from GGUF file and filename heuristics.
    Auto-configures LazyMoE settings optimally for detected model.
    """

    def __init__(self, ram_budget_gb: float = 8.0):
        self.ram_budget_gb = ram_budget_gb

    def detect(self, model_path: str) -> ModelConfig:
        """Main entry point — returns fully configured ModelConfig."""
        if not os.path.exists(model_path):
            logger.warning(f"Model not found: {model_path}")
            return self._default_config(model_path)

        # Try GGUF metadata first (most accurate)
        config = self._read_gguf_metadata(model_path)

        # Fall back to filename heuristics
        if config is None:
            config = self._detect_from_filename(model_path)

        # Set file size
        config.model_size_gb = os.path.getsize(model_path) / 1e9
        config.path = model_path

        # Auto-tune settings based on RAM budget and model
        self._tune_settings(config)

        logger.info(f"Detected: {config.summary}")
        return config

    def _read_gguf_metadata(self, path: str) -> Optional[ModelConfig]:
        """
        Parse GGUF file header to extract model metadata.
        GGUF format: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
        """
        try:
            with open(path, "rb") as f:
                # Check magic
                magic = f.read(4)
                if magic != b"GGUF":
                    return None

                version = struct.unpack("<I", f.read(4))[0]
                tensor_count = struct.unpack("<Q", f.read(8))[0]
                metadata_count = struct.unpack("<Q", f.read(8))[0]

                metadata = {}
                for _ in range(min(metadata_count, 500)):
                    try:
                        key = self._read_gguf_string(f)
                        vtype = struct.unpack("<I", f.read(4))[0]
                        value = self._read_gguf_value(f, vtype)
                        metadata[key] = value
                    except Exception:
                        break

            if not metadata:
                return None

            config = ModelConfig()

            # Extract architecture info
            arch = metadata.get("general.architecture", "unknown")
            config.family = str(arch)
            config.name = metadata.get("general.name", "Unknown Model")
            config.quant = str(metadata.get("general.quantization_version", "unknown"))

            # Layer count
            layer_keys = [
                f"{arch}.block_count",
                "llama.block_count",
                "mistral.block_count",
            ]
            for k in layer_keys:
                if k in metadata:
                    config.num_layers = int(metadata[k])
                    break

            # Attention heads
            head_keys = [f"{arch}.attention.head_count", "llama.attention.head_count"]
            for k in head_keys:
                if k in metadata:
                    config.num_heads = int(metadata[k])
                    break

            # KV heads
            kv_keys = [f"{arch}.attention.head_count_kv", "llama.attention.head_count_kv"]
            for k in kv_keys:
                if k in metadata:
                    config.num_kv_heads = int(metadata[k])
                    break

            # Hidden dimension
            dim_keys = [f"{arch}.embedding_length", "llama.embedding_length"]
            for k in dim_keys:
                if k in metadata:
                    config.hidden_dim = int(metadata[k])
                    config.head_dim = config.hidden_dim // max(config.num_heads, 1)
                    break

            # Context length
            ctx_keys = [f"{arch}.context_length", "llama.context_length"]
            for k in ctx_keys:
                if k in metadata:
                    config.context_length = int(metadata[k])
                    break

            # MoE detection
            expert_keys = [f"{arch}.expert_count", "llama.expert_count"]
            for k in expert_keys:
                if k in metadata:
                    config.num_experts = int(metadata[k])
                    config.is_moe = config.num_experts > 1
                    break

            used_expert_keys = [f"{arch}.expert_used_count", "llama.expert_used_count"]
            for k in used_expert_keys:
                if k in metadata:
                    config.experts_per_token = int(metadata[k])
                    break

            config.kv_dim = config.num_kv_heads * config.head_dim

            # Estimate params from file size
            size_gb = os.path.getsize(path) / 1e9
            # Q4_K_M ≈ 4.5 bits/param → params ≈ size_bytes * 8 / 4.5
            config.params_b = round((size_gb * 1e9 * 8) / 4.5 / 1e9, 1)

            logger.info(f"GGUF metadata: arch={arch} layers={config.num_layers} heads={config.num_heads} kv_heads={config.num_kv_heads} moe={config.is_moe}")
            return config

        except Exception as e:
            logger.debug(f"GGUF parse error: {e}")
            return None

    def _read_gguf_string(self, f) -> str:
        length = struct.unpack("<Q", f.read(8))[0]
        if length > 1024:
            f.seek(length, 1)
            return ""
        return f.read(length).decode("utf-8", errors="replace")

    def _read_gguf_value(self, f, vtype: int):
        TYPE_MAP = {
            0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
            4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: None,
            8: ("<Q", 8), 9: ("<q", 8), 10: ("<d", 8),
        }
        if vtype == 7:  # bool
            return bool(struct.unpack("<B", f.read(1))[0])
        if vtype == 8:  # string
            return self._read_gguf_string(f)
        if vtype == 9:  # array
            elem_type = struct.unpack("<I", f.read(4))[0]
            count = struct.unpack("<Q", f.read(8))[0]
            results = []
            for _ in range(min(count, 100)):
                try:
                    results.append(self._read_gguf_value(f, elem_type))
                except Exception:
                    break
            return results
        if vtype in TYPE_MAP and TYPE_MAP[vtype]:
            fmt, size = TYPE_MAP[vtype]
            return struct.unpack(fmt, f.read(size))[0]
        return None

    def _detect_from_filename(self, path: str) -> ModelConfig:
        """Filename heuristic detection."""
        name = os.path.basename(path).lower()

        # Try known model profiles
        for key, profile in KNOWN_MODELS.items():
            if key in name:
                config = ModelConfig(
                    name=profile.name,
                    family=profile.family,
                    params_b=profile.params_b,
                    num_layers=profile.num_layers,
                    num_heads=profile.num_heads,
                    num_kv_heads=profile.num_kv_heads,
                    head_dim=profile.head_dim,
                    hidden_dim=profile.hidden_dim,
                    is_moe=profile.is_moe,
                    num_experts=profile.num_experts,
                    experts_per_token=profile.experts_per_token,
                    path=path,
                )
                # Detect quantization
                config.quant = self._detect_quant(name)
                return config

        # Generic detection from size hints in filename
        return self._detect_from_size_hint(name, path)

    def _detect_from_size_hint(self, name: str, path: str) -> ModelConfig:
        """Detect param count from filename hints like '7b', '70b', '8x7b'."""
        import re
        config = ModelConfig(path=path)

        # MoE pattern: 8x7b, 8x22b
        moe_match = re.search(r"(\d+)x(\d+)b", name)
        if moe_match:
            n_experts = int(moe_match.group(1))
            expert_size = int(moe_match.group(2))
            config.params_b = n_experts * expert_size
            config.is_moe = True
            config.num_experts = n_experts
            config.experts_per_token = 2
            config.name = f"MoE {n_experts}x{expert_size}B"

        # Dense: 7b, 13b, 70b, 180b
        elif param_match := re.search(r"(\d+(?:\.\d+)?)b", name):
            config.params_b = float(param_match.group(1))
            config.name = f"LLM {config.params_b:.0f}B"

        # Scale layers/dims by param count
        config = self._scale_by_params(config)
        config.quant = self._detect_quant(name)
        return config

    def _scale_by_params(self, config: ModelConfig) -> ModelConfig:
        """Estimate architecture from parameter count."""
        p = config.params_b
        if p <= 3:
            config.num_layers, config.num_heads, config.num_kv_heads, config.hidden_dim = 18, 8, 1, 2048
        elif p <= 8:
            config.num_layers, config.num_heads, config.num_kv_heads, config.hidden_dim = 32, 32, 8, 4096
        elif p <= 14:
            config.num_layers, config.num_heads, config.num_kv_heads, config.hidden_dim = 40, 40, 10, 5120
        elif p <= 35:
            config.num_layers, config.num_heads, config.num_kv_heads, config.hidden_dim = 48, 40, 8, 5120
        elif p <= 72:
            config.num_layers, config.num_heads, config.num_kv_heads, config.hidden_dim = 80, 64, 8, 8192
        elif p <= 141:
            config.num_layers, config.num_heads, config.num_kv_heads, config.hidden_dim = 56, 48, 8, 6144
        elif p <= 200:
            config.num_layers, config.num_heads, config.num_kv_heads, config.hidden_dim = 80, 96, 8, 12288
        else:
            config.num_layers, config.num_heads, config.num_kv_heads, config.hidden_dim = 96, 128, 16, 16384

        config.head_dim = config.hidden_dim // max(config.num_heads, 1)
        config.kv_dim = config.num_kv_heads * config.head_dim
        return config

    def _detect_quant(self, name: str) -> str:
        for q in ["q2_k", "q3_k_m", "q4_0", "q4_k_m", "q4_k_s", "q5_k_m",
                  "q6_k", "q8_0", "f16", "f32", "iq2_xs", "iq3_xs", "iq4_xs"]:
            if q in name:
                return q.upper()
        return "Q4_K_M"  # assume common default

    def _tune_settings(self, config: ModelConfig) -> None:
        """Auto-tune LazyMoE settings for the detected model and RAM budget."""
        ram = self.ram_budget_gb
        size = config.model_size_gb

        # Threads: use more for bigger models, leave headroom for OS
        import os as _os
        cpu_count = _os.cpu_count() or 4
        config.recommended_threads = max(2, min(cpu_count - 1, 8))

        # Context: scale down for larger models to fit KV cache in RAM
        kv_per_tok = config.kv_ram_tq_per_token_gb
        attention_ram = 3.2  # always loaded
        expert_ram = 0.5     # cache slots

        available_for_kv = max(0.5, ram - attention_ram - expert_ram)
        max_ctx_from_ram = int(available_for_kv / max(kv_per_tok, 1e-6))
        config.recommended_ctx = max(512, min(max_ctx_from_ram, config.context_length, 8192))

        # Predict tokens
        config.recommended_n_predict = min(512, config.recommended_ctx // 4)

        # Expert cache slots: for MoE models
        if config.is_moe:
            # Each expert slot costs ram proportional to model size / num_experts
            expert_size_gb = (size / config.num_experts) if config.num_experts > 0 else 1.0
            available_for_experts = max(1.0, ram - attention_ram - 1.0)
            config.expert_cache_slots = max(2, min(
                int(available_for_experts / max(expert_size_gb, 0.1)),
                config.num_experts,
                8,
            ))
        else:
            config.expert_cache_slots = 1  # dense models don't need expert cache

        # KV bits: use fewer bits if model is large
        if size > 40:
            config.kv_cache_bits = 2  # aggressive compression for huge models
        elif size > 15:
            config.kv_cache_bits = 3  # TurboQuant default
        else:
            config.kv_cache_bits = 4  # small models can afford higher quality

        logger.info(
            f"Tuned: threads={config.recommended_threads} "
            f"ctx={config.recommended_ctx} "
            f"predict={config.recommended_n_predict} "
            f"expert_slots={config.expert_cache_slots} "
            f"kv_bits={config.kv_cache_bits}"
        )

    def _default_config(self, path: str) -> ModelConfig:
        return ModelConfig(
            name="Unknown Model",
            family="unknown",
            params_b=7,
            path=path,
        )
