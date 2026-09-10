"""Offline evaluation for literature retrieval quality."""

__all__ = ["compute_ranking_metrics", "run_evaluation"]


def compute_ranking_metrics(*args, **kwargs):
    from .evaluator import compute_ranking_metrics as implementation

    return implementation(*args, **kwargs)


def run_evaluation(*args, **kwargs):
    from .evaluator import run_evaluation as implementation

    return implementation(*args, **kwargs)
