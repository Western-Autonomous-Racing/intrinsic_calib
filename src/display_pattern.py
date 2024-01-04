import cv2
import numpy as np
from screeninfo import get_monitors
import math

class DisplayPattern:

    def __init__(self, viewing_diagonal, pixel_pitch=0.000311, square_size=0.025):
        self.viewing_diagonal = viewing_diagonal
        self.pixel_pitch=pixel_pitch
        self.specs = self.get_main_monitor_specs()
        self.square_size = square_size
        self.height = 0
        self.width = 0
        self.pattern_size = self.get_pattern_size()

    def get_pattern_size(self):
        angle = math.atan(self.specs['height'] / self.specs['width'])
        self.height = self.viewing_diagonal * math.sin(angle)
        self.width = self.viewing_diagonal * math.cos(angle)
        pattern_size = (int(self.width/self.square_size), int(self.height/self.square_size))
        return pattern_size     

    def get_main_monitor_specs(self):
        monitors = get_monitors()
        for m in monitors:
            if m.is_primary:
                return {
                    "width": m.width,
                    "height": m.height,
                    "name": m.name
                }
