"""Logging utilities for the core system."""

import logging
import sys
from typing import Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a configured logger instance.
    
    Args:
        name: Optional name for the logger. If None, uses the calling module's name.
        
    Returns:
        Configured logger instance.
    """
    if name is None:
        # Get the calling module's name
        frame = sys._getframe(1)
        name = frame.f_globals.get('__name__', 'unknown')
    
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
        
        # Set default level
        logger.setLevel(logging.INFO)
    
    return logger