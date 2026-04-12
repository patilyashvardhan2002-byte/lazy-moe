"""
LazyMoE System Detector
Detects hardware configuration and recommends which models can run.
Works on Windows, macOS, Linux — no external dependencies beyond psutil.
"""

import os
import sys
import platform
import subprocess
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("lazy-moe.system")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    logger.warning("psutil not installed — install with: pip install psutil")


@dataclass
class GPUInfo:
    name: str = "Unknown"
    vram_gb: float = 0.0
    vendor: str = "unknown"   # nvidia, amd, intel, apple
    is_integrated: bool = True


@dataclass
class SystemInfo:
    # OS
    os_name: str = ""
    os_version: str = ""
    architecture: str = ""

    # CPU
    cpu_name: str = ""
    cpu_cores_physical: int = 0
    cpu_cores_logical: int = 0
    cpu_freq_mhz: float = 0.0
    cpu_vendor: str = ""      # intel, amd, apple

    # RAM
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    ram_used_pct: float = 0.0

    # GPU
    gpus: list = field(default_factory=list)
    has_discrete_gpu: bool = False
    total_vram_gb: float = 0.0

    # Storage (model drive)
    disk_total_gb: float = 0.0
    disk_free_gb: float = 0.0

    # Apple Silicon special case
    is_apple_silicon: bool = False
    unified_memory_gb: float = 0.0  # shared CPU+GPU memory

    @property
    def effective_ram_gb(self) -> float:
        """RAM available for model loading."""
        if self.is_apple_silicon:
            return self.unified_memory_gb * 0.75  # leave 25% for OS
        return self.ram_total_gb * 0.85  # leave 15% for OS

    @property
    def primary_gpu(self) -> Optional[GPUInfo]:
        if not self.gpus:
            return None
        # Prefer discrete over integrated
        discrete = [g for g in self.gpus if not g.is_integrated]
        return discrete[0] if discrete else self.gpus[0]


# Model compatibility database
MODEL_COMPAT = [
    # name, params_b, min_ram_gb, recommended_ram_gb, min_vram_gb, quant, speed_class, notes
    {"name": "Phi-3 Mini 3.8B",      "params": 3.8,  "min_ram": 3,   "rec_ram": 4,   "min_vram": 0, "quant": "Q4_K_M", "speed": "fast",   "family": "phi"},
    {"name": "Mistral 7B",            "params": 7,    "min_ram": 5,   "rec_ram": 6,   "min_vram": 0, "quant": "Q4_K_M", "speed": "fast",   "family": "mistral"},
    {"name": "Llama 3 8B",            "params": 8,    "min_ram": 5,   "rec_ram": 6,   "min_vram": 0, "quant": "Q4_K_M", "speed": "fast",   "family": "llama"},
    {"name": "Gemma 2 9B",            "params": 9,    "min_ram": 6,   "rec_ram": 8,   "min_vram": 0, "quant": "Q4_K_M", "speed": "fast",   "family": "gemma"},
    {"name": "Mistral 12B",           "params": 12,   "min_ram": 8,   "rec_ram": 10,  "min_vram": 0, "quant": "Q4_K_M", "speed": "fast",   "family": "mistral"},
    {"name": "Phi-4 14B",             "params": 14,   "min_ram": 9,   "rec_ram": 12,  "min_vram": 0, "quant": "Q4_K_M", "speed": "medium", "family": "phi"},
    {"name": "Qwen2.5 14B",           "params": 14,   "min_ram": 9,   "rec_ram": 12,  "min_vram": 0, "quant": "Q4_K_M", "speed": "medium", "family": "qwen"},
    {"name": "Mixtral 8x7B",          "params": 47,   "min_ram": 26,  "rec_ram": 32,  "min_vram": 0, "quant": "Q4_K_M", "speed": "medium", "family": "mixtral", "is_moe": True},
    {"name": "Command R 35B",         "params": 35,   "min_ram": 20,  "rec_ram": 24,  "min_vram": 0, "quant": "Q4_K_M", "speed": "medium", "family": "cohere"},
    {"name": "Llama 3 70B",           "params": 70,   "min_ram": 40,  "rec_ram": 48,  "min_vram": 0, "quant": "Q4_K_M", "speed": "slow",   "family": "llama"},
    {"name": "Qwen2.5 72B",           "params": 72,   "min_ram": 42,  "rec_ram": 48,  "min_vram": 0, "quant": "Q4_K_M", "speed": "slow",   "family": "qwen"},
    {"name": "Mixtral 8x22B",         "params": 141,  "min_ram": 65,  "rec_ram": 80,  "min_vram": 0, "quant": "Q4_K_M", "speed": "slow",   "family": "mixtral", "is_moe": True},
    {"name": "Command R+ 104B",       "params": 104,  "min_ram": 60,  "rec_ram": 72,  "min_vram": 0, "quant": "Q4_K_M", "speed": "slow",   "family": "cohere"},
    {"name": "Llama 3 405B",          "params": 405,  "min_ram": 200, "rec_ram": 256, "min_vram": 0, "quant": "Q4_K_M", "speed": "very_slow", "family": "llama"},
    {"name": "DeepSeek V3 671B",      "params": 671,  "min_ram": 350, "rec_ram": 400, "min_vram": 0, "quant": "Q2_K",   "speed": "very_slow", "family": "deepseek", "is_moe": True},
]

SPEED_LABELS = {
    "fast":      "~10–25 tok/s",
    "medium":    "~3–8 tok/s",
    "slow":      "~1–3 tok/s",
    "very_slow": "<1 tok/s",
}


class SystemDetector:

    def detect(self) -> SystemInfo:
        info = SystemInfo()
        self._detect_os(info)
        self._detect_cpu(info)
        self._detect_ram(info)
        self._detect_gpu(info)
        self._detect_disk(info)
        self._detect_apple_silicon(info)
        logger.info(
            f"System: {info.os_name} | {info.cpu_name} | "
            f"RAM {info.ram_total_gb:.1f}GB | "
            f"GPU {info.primary_gpu.name if info.primary_gpu else 'none'}"
        )
        return info

    def get_model_compatibility(self, info: SystemInfo) -> list:
        """
        Returns list of models with compatibility status for this system.
        Status: 'great' | 'ok' | 'slow' | 'mmap' | 'no'
        """
        results = []
        effective_ram = info.effective_ram_gb

        for model in MODEL_COMPAT:
            min_ram = model["min_ram"]
            rec_ram = model["rec_ram"]

            if effective_ram >= rec_ram:
                status = "great"
                note = f"Runs fully in RAM at {SPEED_LABELS[model['speed']]}"
            elif effective_ram >= min_ram:
                status = "ok"
                note = f"Fits in RAM, slightly slower"
            elif effective_ram >= min_ram * 0.5:
                status = "mmap"
                note = "SSD streaming via mmap — slow but works"
            else:
                status = "no"
                note = f"Needs {min_ram}GB RAM minimum"

            results.append({
                **model,
                "status": status,
                "note": note,
                "speed_label": SPEED_LABELS[model["speed"]],
            })

        return results

    # ── OS ────────────────────────────────────────────────────────────────────
    def _detect_os(self, info: SystemInfo):
        info.os_name = platform.system()
        info.os_version = platform.version()
        info.architecture = platform.machine()

    # ── CPU ───────────────────────────────────────────────────────────────────
    def _detect_cpu(self, info: SystemInfo):
        info.cpu_name = platform.processor() or "Unknown CPU"

        if HAS_PSUTIL:
            info.cpu_cores_physical = psutil.cpu_count(logical=False) or 1
            info.cpu_cores_logical  = psutil.cpu_count(logical=True) or 1
            freq = psutil.cpu_freq()
            info.cpu_freq_mhz = freq.max if freq else 0

        # Detect vendor
        cpu_lower = info.cpu_name.lower()
        if "intel" in cpu_lower:
            info.cpu_vendor = "intel"
        elif "amd" in cpu_lower or "ryzen" in cpu_lower:
            info.cpu_vendor = "amd"
        elif "apple" in cpu_lower or info.architecture == "arm64":
            info.cpu_vendor = "apple"
        else:
            info.cpu_vendor = "unknown"

        # Windows: try wmic for better CPU name
        if info.os_name == "Windows" and (not info.cpu_name or info.cpu_name == ""):
            try:
                result = subprocess.run(
                    ["wmic", "cpu", "get", "name"],
                    capture_output=True, text=True, timeout=5
                )
                lines = [l.strip() for l in result.stdout.split("\n") if l.strip() and l.strip() != "Name"]
                if lines:
                    info.cpu_name = lines[0]
            except Exception:
                pass

    # ── RAM ───────────────────────────────────────────────────────────────────
    def _detect_ram(self, info: SystemInfo):
        if HAS_PSUTIL:
            mem = psutil.virtual_memory()
            info.ram_total_gb    = mem.total / 1e9
            info.ram_available_gb = mem.available / 1e9
            info.ram_used_pct    = mem.percent
        else:
            # Fallback for Windows
            if platform.system() == "Windows":
                try:
                    result = subprocess.run(
                        ["wmic", "computersystem", "get", "TotalPhysicalMemory"],
                        capture_output=True, text=True, timeout=5
                    )
                    for line in result.stdout.split("\n"):
                        line = line.strip()
                        if line.isdigit():
                            info.ram_total_gb = int(line) / 1e9
                            break
                except Exception:
                    pass

    # ── GPU ───────────────────────────────────────────────────────────────────
    def _detect_gpu(self, info: SystemInfo):
        gpus = []

        # Try nvidia-smi first (NVIDIA GPU)
        gpus += self._detect_nvidia()

        # Windows: wmic
        if platform.system() == "Windows":
            gpus += self._detect_gpu_windows(gpus)

        # macOS
        if platform.system() == "Darwin":
            gpus += self._detect_gpu_macos()

        # Linux
        if platform.system() == "Linux":
            gpus += self._detect_gpu_linux(gpus)

        info.gpus = gpus
        info.has_discrete_gpu = any(not g.is_integrated for g in gpus)
        info.total_vram_gb = sum(g.vram_gb for g in gpus if not g.is_integrated)

    def _detect_nvidia(self) -> list:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                return []
            gpus = []
            for line in result.stdout.strip().split("\n"):
                if "," in line:
                    parts = line.split(",")
                    name = parts[0].strip()
                    vram_mb = float(parts[1].strip()) if len(parts) > 1 else 0
                    gpus.append(GPUInfo(
                        name=name,
                        vram_gb=round(vram_mb/1024, 1),
                        vendor="nvidia",
                        is_integrated=False,
                    ))
            return gpus
        except Exception:
            return []

    def _detect_gpu_windows(self, existing: list) -> list:
        existing_names = {g.name.lower() for g in existing}
        try:
            result = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "name,AdapterRAM"],
                capture_output=True, text=True, timeout=5
            )
            gpus = []
            for line in result.stdout.split("\n"):
                line = line.strip()
                if not line or "Name" in line or "AdapterRAM" in line:
                    continue
                parts = line.split()
                if not parts:
                    continue

                # Last token might be VRAM in bytes
                vram_gb = 0.0
                name_parts = parts
                if parts[-1].isdigit() and len(parts[-1]) > 6:
                    vram_gb = round(int(parts[-1]) / 1e9, 1)
                    name_parts = parts[:-1]

                name = " ".join(name_parts)
                if not name or name.lower() in existing_names:
                    continue

                name_lower = name.lower()
                vendor = "nvidia" if "nvidia" in name_lower else \
                         "amd" if ("amd" in name_lower or "radeon" in name_lower) else \
                         "intel" if "intel" in name_lower else "unknown"
                is_integrated = "intel" in name_lower or "uhd" in name_lower or "iris" in name_lower

                gpus.append(GPUInfo(name=name, vram_gb=vram_gb, vendor=vendor, is_integrated=is_integrated))
            return gpus
        except Exception:
            return []

    def _detect_gpu_macos(self) -> list:
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                capture_output=True, text=True, timeout=10
            )
            import json
            data = json.loads(result.stdout)
            gpus = []
            for display in data.get("SPDisplaysDataType", []):
                name = display.get("sppci_model", display.get("_name", "Unknown GPU"))
                vram_str = display.get("spdisplays_vram", "0")
                vram_gb = 0.0
                if "GB" in vram_str:
                    vram_gb = float(vram_str.replace("GB","").strip())
                elif "MB" in vram_str:
                    vram_gb = float(vram_str.replace("MB","").strip()) / 1024
                vendor = "apple" if "apple" in name.lower() else \
                         "nvidia" if "nvidia" in name.lower() else \
                         "amd" if "amd" in name.lower() else "intel"
                gpus.append(GPUInfo(name=name, vram_gb=vram_gb, vendor=vendor, is_integrated=True))
            return gpus
        except Exception:
            return []

    def _detect_gpu_linux(self, existing: list) -> list:
        try:
            result = subprocess.run(
                ["lspci"], capture_output=True, text=True, timeout=5
            )
            gpus = []
            for line in result.stdout.split("\n"):
                if "VGA" in line or "3D" in line or "Display" in line:
                    name_lower = line.lower()
                    vendor = "nvidia" if "nvidia" in name_lower else \
                             "amd" if ("amd" in name_lower or "radeon" in name_lower) else \
                             "intel" if "intel" in name_lower else "unknown"
                    is_integrated = vendor == "intel"
                    name = line.split(":")[-1].strip()
                    gpus.append(GPUInfo(name=name, vendor=vendor, is_integrated=is_integrated))
            return gpus
        except Exception:
            return []

    # ── Disk ─────────────────────────────────────────────────────────────────
    def _detect_disk(self, info: SystemInfo):
        if HAS_PSUTIL:
            try:
                usage = psutil.disk_usage(os.path.expanduser("~"))
                info.disk_total_gb = usage.total / 1e9
                info.disk_free_gb  = usage.free / 1e9
            except Exception:
                pass

    # ── Apple Silicon ─────────────────────────────────────────────────────────
    def _detect_apple_silicon(self, info: SystemInfo):
        if platform.system() != "Darwin":
            return
        if platform.machine() != "arm64":
            return

        info.is_apple_silicon = True
        info.cpu_vendor = "apple"

        # Unified memory = RAM total on Apple Silicon
        info.unified_memory_gb = info.ram_total_gb

        # Update GPU to reflect unified memory
        for gpu in info.gpus:
            if gpu.vendor == "apple":
                gpu.vram_gb = info.ram_total_gb  # unified
                gpu.is_integrated = False  # Apple GPU is genuinely fast

        # Try to get chip name
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=3
            )
            if result.returncode == 0:
                info.cpu_name = result.stdout.strip()
        except Exception:
            pass
