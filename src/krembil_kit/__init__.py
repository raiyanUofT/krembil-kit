__version__ = "1.0.3"

import logging

from .edf_loader import EDFLoader
from .trigger_detector import TriggerDetector
from .spectral_analyzer import SpectralAnalyzer
from .connectivity_analyzer import ConnectivityAnalyzer

def set_log_level(level='INFO'):
    """
    Set logging level for krembil_kit package.
    
    Parameters:
    -----------
    level : str or int
        Logging level. Can be:
        - 'CRITICAL' or logging.CRITICAL (50): Only progress/milestone messages
        - 'ERROR' or logging.ERROR (40): Error messages  
        - 'WARNING' or logging.WARNING (30): Warning messages
        - 'INFO' or logging.INFO (20): General information (default)
        - 'DEBUG' or logging.DEBUG (10): Detailed debugging information
        
    Examples:
    ---------
    >>> import krembil_kit
    >>> krembil_kit.set_log_level('CRITICAL')  # Show critical-level (and above) log statements
    >>> krembil_kit.set_log_level('INFO')      # Show info-level (and above) log statements
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    logger = logging.getLogger('krembil_kit')
    logger.setLevel(level)
    
    # Add handler if none exists
    if not logger.handlers:
        handler = logging.StreamHandler()
        # Clean format - just the message (like print statements)
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Prevent propagation to root logger to avoid duplicate messages
        logger.propagate = False

__all__ = ["EDFLoader", 
           "TriggerDetector",
           "SpectralAnalyzer",
           "ConnectivityAnalyzer",
           "set_log_level"]