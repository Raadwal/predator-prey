from dataclasses import dataclass
from typing import Tuple

def lighten_color(color: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
    if not (0 < factor <= 1):
        raise ValueError("Factor should be greater than 0 and less than or equal to 1")
    
    return tuple(min(int(c + (255 - c) * factor), 255) for c in color)

def darken_color(color: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
    if not (0 < factor <= 1):
        raise ValueError("Factor should be greater than 0 and less than or equal to 1")
    
    return tuple(max(int(c * (1 - factor)), 0) for c in color)

@dataclass
class Colors:
    BACKGROUND_COLOR: Tuple[int, int, int] = (34, 139, 34)
    WATER_SOURCE_COLOR: Tuple[int, int, int] = (64, 164, 223)

    PREY_COLOR: Tuple[int, int, int] = (169, 169, 169)
    PREY_COLLIDER_COLOR: Tuple[int, int, int] = lighten_color(PREY_COLOR, 0.3)

    PREDATOR_COLOR: Tuple[int, int, int] = (255, 89, 0)
    PREDATOR_COLIDER_COLOR: Tuple[int, int, int] = lighten_color(PREDATOR_COLOR, 0.3)

    PLANT_COLOR: Tuple[int, int, int] = (0, 100, 0)
    PLANT_EATEN_COLOR: Tuple[int, int, int] = (139, 69, 19)