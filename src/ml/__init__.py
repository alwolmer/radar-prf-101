from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import (
    BaseFeaturizationRun,
    BaseMLflowRegressionExperiment,
    BaseModelServingEndpoint,
)

if TYPE_CHECKING:
    from .activity_group_regression import (
        ActivityGroupExperimentConfig,
        ActivityGroupFeaturizationConfig,
        ActivityGroupFeaturizationRun,
        ActivityGroupRegressionExperiment,
    )

__all__ = [
    "BaseFeaturizationRun",
    "BaseMLflowRegressionExperiment",
    "BaseModelServingEndpoint",
    "ActivityGroupFeaturizationRun",
    "ActivityGroupRegressionExperiment",
    "ActivityGroupExperimentConfig",
    "ActivityGroupFeaturizationConfig",
]


def __getattr__(name: str) -> Any:
    if name in {
        "ActivityGroupFeaturizationRun",
        "ActivityGroupRegressionExperiment",
        "ActivityGroupExperimentConfig",
        "ActivityGroupFeaturizationConfig",
    }:
        from . import activity_group_regression as activity_group

        return getattr(activity_group, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
