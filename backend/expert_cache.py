"""
LazyMoE Expert Cache Manager
LRU cache for MoE expert weight shards loaded from SSD into RAM.
"""

import os
import time
import mmap
import threading
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

logger = logging.getLogger("lazy-moe.cache")


@dataclass
class ExpertShard:
    expert_id: int
    size_bytes: int
    loaded_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    hits: int = 0
    data: Optional[np.ndarray] = None  # actual weight bytes when loaded

    @property
    def size_gb(self) -> float:
        return self.size_bytes / 1e9

    @property
    def age_ms(self) -> float:
        return (time.time() - self.loaded_at) * 1000


class ExpertCache:
    """
    Thread-safe LRU cache for expert weight shards.
    Experts are stored on SSD as GGUF shards and loaded on demand.
    """

    def __init__(self, capacity: int, shard_dir: str, ram_budget_gb: float = 4.0):
        self.capacity = capacity
        self.shard_dir = shard_dir
        self.ram_budget_gb = ram_budget_gb
        self._cache: OrderedDict[int, ExpertShard] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0, "bytes_loaded": 0}
        self._prefetch_queue: set[int] = set()

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, expert_id: int) -> Optional[ExpertShard]:
        """Retrieve expert from cache (cache hit) or load from SSD (miss)."""
        with self._lock:
            if expert_id in self._cache:
                return self._hit(expert_id)
            return self._miss(expert_id)

    def prefetch(self, expert_ids: list[int]) -> None:
        """Background-load experts predicted to be needed soon."""
        def _load():
            for eid in expert_ids:
                with self._lock:
                    if eid not in self._cache and eid not in self._prefetch_queue:
                        self._prefetch_queue.add(eid)
                        self._load_from_ssd(eid)
                        self._prefetch_queue.discard(eid)

        thread = threading.Thread(target=_load, daemon=True)
        thread.start()

    def evict(self, expert_id: int) -> None:
        """Manually evict a specific expert from cache."""
        with self._lock:
            if expert_id in self._cache:
                del self._cache[expert_id]
                logger.debug(f"Manually evicted expert {expert_id}")

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            logger.info("Cache cleared")

    # ── Stats & Inspection ────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            return {
                **self._stats,
                "hit_rate": self._stats["hits"] / total if total > 0 else 0.0,
                "cached_experts": list(self._cache.keys()),
                "ram_used_gb": self.ram_used_gb,
                "capacity": self.capacity,
            }

    @property
    def ram_used_gb(self) -> float:
        with self._lock:
            return sum(s.size_bytes for s in self._cache.values()) / 1e9

    @property
    def cached_ids(self) -> list[int]:
        with self._lock:
            return list(self._cache.keys())

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "expert_id": eid,
                    "size_gb": round(s.size_gb, 3),
                    "hits": s.hits,
                    "age_ms": round(s.age_ms),
                    "lru_rank": i,
                }
                for i, (eid, s) in enumerate(self._cache.items())
            ]

    # ── Internals ─────────────────────────────────────────────────────────────

    def _hit(self, expert_id: int) -> ExpertShard:
        self._cache.move_to_end(expert_id)
        shard = self._cache[expert_id]
        shard.last_used = time.time()
        shard.hits += 1
        self._stats["hits"] += 1
        logger.debug(f"Cache HIT  expert={expert_id} hits={shard.hits}")
        return shard

    def _miss(self, expert_id: int) -> ExpertShard:
        self._stats["misses"] += 1
        logger.debug(f"Cache MISS expert={expert_id} — loading from SSD")
        return self._load_from_ssd(expert_id)

    def _load_from_ssd(self, expert_id: int) -> ExpertShard:
        """Load expert weights from SSD shard file into RAM."""
        shard_path = self._shard_path(expert_id)

        if not os.path.exists(shard_path):
            # In simulation mode: create a dummy shard
            logger.warning(f"Shard not found: {shard_path} — using dummy weights")
            return self._make_dummy_shard(expert_id)

        t0 = time.perf_counter()
        size = os.path.getsize(shard_path)

        # Use mmap for zero-copy load — OS handles paging from SSD
        with open(shard_path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            data = np.frombuffer(mm, dtype=np.uint8).copy()
            mm.close()

        elapsed_ms = (time.perf_counter() - t0) * 1000
        mb_per_sec = (size / 1e6) / (elapsed_ms / 1000)

        shard = ExpertShard(expert_id=expert_id, size_bytes=size, data=data)
        self._stats["bytes_loaded"] += size

        logger.info(
            f"Loaded expert={expert_id} size={size/1e6:.1f}MB "
            f"time={elapsed_ms:.0f}ms speed={mb_per_sec:.0f}MB/s"
        )

        self._insert(expert_id, shard)
        return shard

    def _insert(self, expert_id: int, shard: ExpertShard) -> None:
        """Insert shard into cache, evicting LRU if needed."""
        while len(self._cache) >= self.capacity:
            evicted_id, evicted = self._cache.popitem(last=False)
            self._stats["evictions"] += 1
            logger.debug(f"Evicted expert={evicted_id} (LRU, hits={evicted.hits})")

        self._cache[expert_id] = shard
        self._cache.move_to_end(expert_id)

    def _shard_path(self, expert_id: int) -> str:
        return os.path.join(self.shard_dir, f"expert_{expert_id:02d}.bin")

    def _make_dummy_shard(self, expert_id: int) -> ExpertShard:
        """Simulate a 350MB 1-bit expert shard for testing without real weights."""
        DUMMY_SIZE = 350 * 1024 * 1024  # 350MB ≈ 1-bit expert
        # Don't actually allocate — just simulate timing
        load_time = DUMMY_SIZE / (3.5e9)  # simulate NVMe at 3.5 GB/s
        time.sleep(load_time * 0.1)       # 10% of real time for demo
        shard = ExpertShard(expert_id=expert_id, size_bytes=DUMMY_SIZE)
        self._stats["bytes_loaded"] += DUMMY_SIZE
        self._insert(expert_id, shard)
        return shard
