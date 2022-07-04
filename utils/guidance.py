import numpy as np
import torch
from abc import ABC, abstractmethod

class GuidanceMapping(ABC):
    @abstractmethod
    def compute(self,x,g,size):
        return NotImplemented

    @property
    def _channel(self):
        return 1

class CustomGuidanceMapping(GuidanceMapping):
    '''
        User-defined guidance mapping
    '''
    def compute(self,x,g,size):
        return NotImplemented

class ModXMapping(GuidanceMapping):
    def compute(self,x,g,size):
        cr, h, w = size    
        coords_x = g[:,1]
        mod_x = torch.clamp(coords_x / w, 0.0, 1.0)
        return mod_x.unsqueeze(1)

class ModYMapping(GuidanceMapping):
    def compute(self,x,g,size):
        cr, h, w = size
        coords_y = g[:,0]
        mod_y = torch.clamp(coords_y / h, 0.0, 1.0)
        return mod_y.unsqueeze(1)
