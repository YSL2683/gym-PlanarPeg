import os
import re
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def get_latest_checkpoint(checkpoint_dir):
    """Get latest checkpoint file (step_*.pth format)."""
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None

    step_files = list(checkpoint_dir.glob("step_*.pth"))
    if not step_files:
        logger.warning(f"No step_*.pth files found in {checkpoint_dir}")
        return None

    step_numbers = []
    for step_file in step_files:
        match = re.search(r'step_(\d+)', step_file.name)
        if match:
            step_numbers.append((int(match.group(1)), step_file))

    if not step_numbers:
        return None

    latest_step, latest_file = max(step_numbers, key=lambda x: x[0])
    logger.info(f"Found latest checkpoint: {latest_file.name} (step {latest_step})")
    return str(latest_file)

def get_best_checkpoint(checkpoint_dir):
    """Get best checkpoint file."""
    checkpoint_dir = Path(checkpoint_dir)
    best_file = checkpoint_dir / "best.pth"
    
    if best_file.exists():
        logger.info(f"Found best checkpoint: {best_file.name}")
        return str(best_file)
        
    return None
