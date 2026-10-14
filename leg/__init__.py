"""Humanoid leg load calculator: physics only (no UI imports).  See README.md for the model and equations.

Typical use::

    cfg = load_config("leg_config.yaml")
    cfg = with_actuator_choices(cfg, {"ankle_motor": "X6-50-L"})       # optional hardware swap
    p = cfg.presets["deep_squat"]
    r = analyze(cfg, Inputs(q=p.q, load=default_load(cfg, p.load_fraction)))
    r.static.tau, r.actuators, r.links, r.flanges, r.warnings
"""
from .analysis import Inputs, Results, analyze  # noqa: F401
from .config import Config, ConfigError, load_config, with_actuator_choices  # noqa: F401
from .statics import Load, default_load  # noqa: F401
