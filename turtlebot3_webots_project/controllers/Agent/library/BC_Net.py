import os
import torch as th
import torch.nn as nn

from typing import Optional

class modelNet(nn.Module):
    def __init__(self, 
                 observation_shape:int, 
                 action_shape:int):
        super(modelNet, self).__init__()
        layer = [nn.Linear(observation_shape, 256),
                 nn.ReLU(),
                 nn.Linear(256, 256),
                 nn.ReLU(),
                 nn.Dropout(p=0.2),
                 nn.Linear(256, 256),
                 nn.ReLU(),
                 nn.Dropout(p=0.2),
                 nn.Linear(256, action_shape),
                 nn.Tanh()]
        
        self.layer = nn.Sequential(*layer)

    def forward(self, observation):
        return self.layer(observation)
    
    def save_model(self, 
                   path: str,
                   name: str):
        """Save model to a specified path."""
        last_path = str(os.path.join(path, name))+f".pth"
        if os.path.exists(path):
            pass
        else:
            os.makedirs(path, exist_ok=True)
        th.save(self.state_dict(), last_path)
        print(f"Model saved to {path}")

    def load_model(self, 
                   path: str, 
                   name:str,
                   device:str="cuda"):
        """Load model from a specified path."""
        path = str(os.path.join(path, name))+f".pth"
        self.load_state_dict(th.load(path, 
                                        map_location=device,
                                        weights_only=True))
        print( f"Model loaded from {path}")
