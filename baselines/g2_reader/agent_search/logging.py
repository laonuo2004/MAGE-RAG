import logging
from pathlib import Path

def setup_logging(save_dir: str, item: dict = None):
    """Setup process-specific logging system"""
    if item is not None:
        log_dir = Path(save_dir) / "logs"/ f"data_{item['_id']}" 
        log_dir.mkdir(parents=True, exist_ok=True)
    else:
        log_dir = Path(save_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
    
    if item is not None:
        log_file = log_dir / f"process_{item['_id']}.log"
        logger_name = f"process_{item['_id']}"  
    else:
        log_file = log_dir / "main.log"
        logger_name = "main"
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        if item is not None:
            formatter = logging.Formatter(
                f'[Process-{item["_id"]}] %(asctime)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger