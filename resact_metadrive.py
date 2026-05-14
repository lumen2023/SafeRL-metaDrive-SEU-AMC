"""Backward-compatible ResAct import path."""

from safe_metadrive_adapter.resact import RESACT_INFO_KEYS, ResidualActionWrapper, wrap_residual_action_env

ResActWrapper = ResidualActionWrapper

__all__ = [
    "RESACT_INFO_KEYS",
    "ResidualActionWrapper",
    "ResActWrapper",
    "wrap_residual_action_env",
]
