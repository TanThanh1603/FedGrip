"""Configuration for controlled pretrained-backbone FedAvg baselines."""
from dataclasses import dataclass, fields
import math


@dataclass(frozen=True)
class BackboneControlConfig:
    """Shared controls for LP-FT, L2-SP, and TGBA comparisons."""

    linear_probe_ratio: float = 0.1
    l2sp_weight: float = 1e-4
    selection_margin: float = 0.01
    max_relative_drift: float = 0.2

    def validate(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
        if not 0 <= self.linear_probe_ratio < 1:
            raise ValueError("linear_probe_ratio must be in [0, 1)")
        if self.l2sp_weight < 0:
            raise ValueError("l2sp_weight must be non-negative")
        if not 0 <= self.selection_margin < 1:
            raise ValueError("selection_margin must be in [0, 1)")
        if self.max_relative_drift <= 0:
            raise ValueError("max_relative_drift must be positive")

    @classmethod
    def from_args(cls, args):
        config = cls(**{
            field.name: getattr(args, "control_" + field.name, field.default)
            for field in fields(cls)
        })
        config.validate()
        return config
