from backend.learning.engine import LearningEngine

_engine_instance = None


def get_learning_engine(config_path: str = "config.yaml") -> LearningEngine:
    """Singleton accessor for LearningEngine"""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = LearningEngine(config_path)
    else:
        # mtime-gated: a single stat() unless config.yaml was saved since last use
        _engine_instance.reload_config_if_changed()
    return _engine_instance
