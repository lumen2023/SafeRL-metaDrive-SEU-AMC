"""Portable SafeMetaDrive adapter package.

The package keeps imports lazy so lightweight users can import the risk-field or
ResAct helpers without importing the full MetaDrive runtime.
"""

__all__ = [
    "DEFAULT_CONFIG",
    "TRAINING_CONFIG",
    "VALIDATION_CONFIG",
    "SafeMetaDriveAdapterEnv",
    "SafeMetaDriveEnv_mini",
    "get_training_env",
    "get_validation_env",
    "make_env_factory",
    "make_safe_metadrive_env",
    "RESACT_INFO_KEYS",
    "ResidualActionWrapper",
    "wrap_residual_action_env",
    "RiskFieldCalculator",
    "prefer_local_metadrive",
]


def __getattr__(name):
    if name in {"DEFAULT_CONFIG", "TRAINING_CONFIG", "VALIDATION_CONFIG"}:
        from . import config

        return getattr(config, name)
    if name in {"RiskFieldCalculator"}:
        from .risk_field import RiskFieldCalculator

        return RiskFieldCalculator
    if name in {"prefer_local_metadrive"}:
        from .local_import import prefer_local_metadrive

        return prefer_local_metadrive
    if name in {"RESACT_INFO_KEYS", "ResidualActionWrapper", "wrap_residual_action_env"}:
        from . import resact

        return getattr(resact, name)
    if name in {"SafeMetaDriveAdapterEnv", "SafeMetaDriveEnv_mini"}:
        from . import env

        return getattr(env, name)
    if name in {"get_training_env", "get_validation_env", "make_env_factory", "make_safe_metadrive_env"}:
        from . import factory

        return getattr(factory, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
