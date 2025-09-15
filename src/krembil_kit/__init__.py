__version__ = "1.0.2"

from .edf_loader import EDFLoader
from .trigger_detector import TriggerDetector
from .spectral_analyzer import SpectralAnalyzer
from .connectivity_analyzer import ConnectivityAnalyzer

__all__ = ["EDFLoader", 
           "TriggerDetector",
           "SpectralAnalyzer",
           "ConnectivityAnalyzer"]