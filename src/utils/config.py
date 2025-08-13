import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from .logger import get_logger

logger = get_logger(__name__)


class ConfigManager:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self):
        try:
            if not self.config_path.exists():
                logger.error(f"Configuration file not found: {self.config_path}")
                raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as file:
                self.config = yaml.safe_load(file)
            
            logger.info(f"Configuration loaded successfully from {self.config_path}")
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading configuration: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            logger.debug(f"Configuration key '{key}' not found, using default: {default}")
            return default
    
    def get_model_config(self) -> Dict[str, Any]:
        return self.config.get('model', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        return self.config.get('data', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        return self.config.get('training', {})
    
    def get_hardware_config(self) -> Dict[str, Any]:
        return self.config.get('hardware', {})
    
    def get_inference_config(self) -> Dict[str, Any]:
        return self.config.get('inference', {})
    
    def validate_config(self) -> bool:
        required_sections = ['model', 'data', 'training', 'hardware', 'inference']
        
        for section in required_sections:
            if section not in self.config:
                logger.error(f"Missing required configuration section: {section}")
                return False
        
        # Validate model configuration
        model_config = self.config['model']
        if 'input_size' not in model_config or 'num_classes' not in model_config:
            logger.error("Model configuration missing required fields")
            return False
        
        # Validate data configuration
        data_config = self.config['data']
        if 'fast_inference_subset' not in data_config:
            logger.error("Data configuration missing fast_inference_subset")
            return False
        
        logger.info("Configuration validation passed")
        return True
    
    # def print_config(self):
    #     logger.info("Current Configuration:")
    #     logger.info(f"Model: {self.config.get('model', {}).get('name', 'Unknown')}")
    #     logger.info(f"Input Size: {self.config.get('model', {}).get('input_size', 'Unknown')}")
    #     logger.info(f"Classes: {self.config.get('model', {}).get('num_classes', 'Unknown')}")
    #     logger.info(f"Fast Inference Subset: {self.config.get('data', {}).get('fast_inference_subset', 'Unknown')}")


# Global configuration instance
config_manager = ConfigManager()
