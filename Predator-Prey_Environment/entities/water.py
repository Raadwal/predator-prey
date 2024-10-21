from typing import Tuple

from utils.vector_2d import Vector2D
from .entity import Entity

class Water(Entity):

    COLOR: Tuple[int, int, int] = (64, 164, 223)

    def __init__(
        self, 
        id: int,
        position: Vector2D,
        entity_radius: float,
        collider_radius: float,
        render_collision_outline: bool = False
    ) -> None:
        
        super().__init__(
            id,
            position,
            entity_radius,
            collider_radius,
            self.COLOR,
            render_collision_outline,
            )