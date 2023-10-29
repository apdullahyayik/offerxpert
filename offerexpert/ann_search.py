"""Module for ANN search."""
# pylint: disable=too-many-instance-attributes,no-member
from math import sqrt
from typing import Optional, Tuple

import faiss
import numpy as np
from faiss import omp_set_num_threads, write_index


class Indexer:
    """Indexer class."""

    __slots__ = (
        "num_gpu",
        "reduce_memory",
        "d",
        "n",
        "nlist",
        "n_probe",
        "_index",
        "num_thread",
        "brute_force",
        "distance_metric",
        "_abbrev",
    )

    def __init__(
        self,
        num_gpu: int | None,
        reduce_memory: bool = False,
        brute_force: bool = False,
        **kwargs,
    ):
        """Construct."""
        self.brute_force = brute_force
        self.reduce_memory: bool = reduce_memory
        self.num_gpu = 0 if num_gpu is None else num_gpu
        self.d: Optional[int] = None
        self.n: Optional[int] = None
        self.n: Optional[int] = None
        self.nlist: Optional[int] = None
        self.n_probe: Optional[int] = None
        self._index = None
        self._abbrev = {"InnerProduct": "IP"}
        self.num_thread: int | None = kwargs.get("num_thread")
        self.distance_metric: str | None = kwargs.get("distance_metric")

    def __call__(self, item_vectors: np.ndarray):
        """Create `Faiss` indexer."""
        # For efficiency use 32-bit float numbers
        # item_vectors.astype(np.float32)

        # Normalize vectors
        self._normalize(item_vectors)

        # n database size, d dimension
        self.n, self.d = item_vectors.shape
        self.nlist = int(4 * sqrt(self.n))
        self.n_probe = max(self.nlist // 32, 1)

        res = None
        if self.num_gpu == 1:
            res = faiss.StandardGpuResources()  # type: ignore
            res.setTempMemory((1024 * 3) * 15)
        elif self.num_gpu > 1:
            res = faiss.StandardGpuResources()  # type: ignore
            res.setTempMemory((1024 * 3) * 15)
            self.num_gpu = 1

        quantizer_method = self._get_quantizer_method()
        quantizer = quantizer_method(self.d)

        if not self.brute_force:
            metric = self._get_metric()

            if self.reduce_memory:
                # Each sub-vector is encoded as m bits
                m = 16
                assert self.d % m == 0, "Mismatch"
                self._index = faiss.IndexIVFFlat(
                    quantizer,
                    self.d,
                    self.nlist if self.n > 100 else 2,  # for local support
                    m,
                    8,
                    metric,
                )
            else:
                self._index = faiss.IndexIVFFlat(
                    quantizer,
                    self.d,
                    self.nlist if self.n > 100 else 2,  # for local support
                    metric,
                )
            self._index.nprobe = self.n_probe
            assert self._index.is_trained is False
        else:
            self._index = quantizer

        self._index.train(item_vectors)  # type: ignore
        assert self._index.is_trained is True, "Index should be trained"

        if self.num_gpu >= 1:
            self._cpu_to_gpu(res)

        self._index.add(item_vectors)  # type: ignore

    @staticmethod
    def _normalize(item_vectors: np.array):  # type: ignore
        faiss.normalize_L2(item_vectors)

    def __repr__(self) -> str:
        """Return string representation of the object."""
        return (
            f"Indexer(num_gpu={self.num_gpu}, "
            f"reduce_memory={self.reduce_memory}, "
            f"num_thread={self.num_thread})"
        )

    def _get_metric(self):
        return getattr(
            faiss, f"METRIC_{self.distance_metric.upper()}", faiss.IndexFlatL2  # type: ignore
        )

    def _get_quantizer_method(self):
        return getattr(
            faiss,
            f"IndexFlat"
            f"{self._abbrev.get(self.distance_metric, self.distance_metric)}",  # type: ignore
            faiss.IndexFlatL2,
        )

    def _cpu_to_gpu(self, res=None):
        assert self.num_gpu >= 1, "GPU does not detected"
        if self.num_gpu == 1:
            self._index = faiss.index_cpu_to_gpu(res, 0, self._index)  # type: ignore
        elif self.num_gpu > 1:
            self._index = faiss.index_cpu_to_all_gpus(self._index)

    def _gpu_to_cpu(self):
        self._index = faiss.index_gpu_to_cpu(self._index)  # type: ignore

    def save(self, path: str):
        """Save indexer."""
        if self.num_gpu >= 1:
            self._gpu_to_cpu()
        write_index(self._index, path)

    def load(self, path: str):
        """Load indexer."""
        self._index = faiss.read_index(path)
        if self.num_gpu >= 1:
            self._cpu_to_gpu()
        self.n = self._index.ntotal
        self.d = self._index.d
        self.nlist = self._index.nlist
        self.n_probe = self._index.n_probe

    def search(
        self, query: np.ndarray, top_n: int, num_thread: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search vector in the indexer."""
        if isinstance(self.num_thread, int):
            num_thread = self.num_thread
        omp_set_num_threads(num_thread)
        return self._index.search(query, top_n)  # type: ignore

    def add(self, vector: np.ndarray):
        """Add vector to the indexer."""
        self._index.add(vector)  # type: ignore

    def index_name(self) -> str:
        """Return name of the index."""
        return self._index.__class__.__name__
