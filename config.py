# config.py
import yaml
from pathlib import Path
from model import build_model

class Config:
    def __init__(self, params_file='params.yaml'):
        self.params_file = params_file
        self.load_params()
        self.model = build_model()
    
    def load_params(self):
        with open(self.params_file, 'r') as f:
            params = yaml.safe_load(f)
        
        # Add default hyperparameters if not present
        if 'hyp' not in params:
            params['hyp'] = {
                'lr': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'giou': 0.05,
                'cls': 0.5,
                'obj': 1.0,
                'anchor_t': 4.0,
                'gr': 1.0
            }
        
        self.MODEL = ConfigDict(params['MODEL'])
        self.TRAIN = ConfigDict(params['TRAIN'])
        self.VALIDATION = ConfigDict(params['VALIDATION'])
        self.hyp = params['hyp']
    
    def __getattr__(self, attr):
        return getattr(self, attr.upper())

class ConfigDict(dict):
    def __init__(self, config_dict):
        super().__init__()
        for k, v in config_dict.items():
            setattr(self, k, v)

    def __getattr__(self, attr):
        return self[attr]

# Create an instance of Config and expose it as a module-level variable
CONFIG = Config()

# Expose MODEL, TRAIN, VALIDATION as module-level variables
MODEL = CONFIG.MODEL
TRAIN = CONFIG.TRAIN
VALIDATION = CONFIG.VALIDATION
MODEL_INSTANCE = CONFIG.model