from text_features_detector.strategies.geval_strategy import run_geval
from text_features_detector.strategies.simple_binary import run_simple_binary

STRATEGY_MAP = {
    "simple_binary": run_simple_binary,
    "geval": run_geval,
}

__all__ = ["run_simple_binary", "run_geval", "STRATEGY_MAP"]
