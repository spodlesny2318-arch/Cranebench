from .kaimal import KaimalParams, KaimalWind, kaimal_psd, synthesise
from .dryden import DrydenParams, DrydenWind

WINDS = {"kaimal": KaimalWind, "dryden": DrydenWind}
WIND_PARAMS = {"kaimal": KaimalParams, "dryden": DrydenParams}

__all__ = ["KaimalParams", "KaimalWind", "kaimal_psd", "synthesise",
           "DrydenParams", "DrydenWind", "WINDS", "WIND_PARAMS"]
