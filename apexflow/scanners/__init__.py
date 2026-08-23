from .base import BaseScanner
from .options_flow import OptionsFlowScanner
from .pre_breakout import PreBreakoutScanner
from .momentum_ignition import MomentumIgnitionScanner
from .short_squeeze import ShortSqueezeScanner
from .earnings_alpha import EarningsAlphaScanner
from .squeeze_radar import SqueezeRadarScanner

ALL_SCANNERS = {
    "options-flow":     OptionsFlowScanner,
    "pre-breakout":     PreBreakoutScanner,
    "momentum":         MomentumIgnitionScanner,
    "squeeze":          ShortSqueezeScanner,
    "earnings":         EarningsAlphaScanner,
    "radar":            SqueezeRadarScanner,
}

__all__ = [
    "BaseScanner",
    "OptionsFlowScanner",
    "PreBreakoutScanner",
    "MomentumIgnitionScanner",
    "ShortSqueezeScanner",
    "EarningsAlphaScanner",
    "SqueezeRadarScanner",
    "ALL_SCANNERS",
]
