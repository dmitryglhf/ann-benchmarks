from __future__ import absolute_import

import numpy as np
from edda import IndexFlat

from ..base.module import BaseANN


class Edda(BaseANN):
    def __init__(self, metric, **kwargs):
        self.metric = metric
        self.index = None
        self.dim = None

        # Edda поддерживает "l2" и "cosine"
        if metric == "euclidean":
            self.edda_metric = "euclid"
        elif metric in ["angular", "cosine"]:
            self.edda_metric = "cosine"
        else:
            raise NotImplementedError(f"Metric {metric} not supported by Edda")

    def fit(self, X):
        self.dim = X.shape[1]
        self.index = IndexFlat(dim=self.dim, metric=self.edda_metric)

        # Edda требует int64 ids
        ids = np.arange(len(X), dtype=np.int64)
        self.index.add(X.astype(np.float32), ids)

    def query(self, v, k):
        v = np.array([v], dtype=np.float32)
        result = self.index.search(v, k)
        return result.ids[0].tolist()

    def set_query_arguments(self, **kwargs):
        pass

    def __str__(self):
        return f"Edda(metric={self.metric})"
