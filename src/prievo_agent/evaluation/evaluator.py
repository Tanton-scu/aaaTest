"""兼容 import facade；产品路径始终实际执行 Candidate heuristic。"""

from .executor import ExecutableDatasetEvaluator


class DatasetEvaluator(ExecutableDatasetEvaluator):
    """保留历史类名，不再使用 code-hash shuffle 伪评价。"""


__all__ = ["DatasetEvaluator"]
