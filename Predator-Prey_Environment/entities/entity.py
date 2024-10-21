import pygame

from typing import Tuple

from utils import Vector2D
from colliders import CircleCollider

class Entity:

    def __init__(
        self, 
        id: int,
        position: Vector2D,
        entity_radius: float,
        collider_radius: float,
        color: Tuple[int, int, int],
        render_collision_outline: bool = False
    ) -> None:

        self._id: int = id
        self._position: Vector2D = position
        self._radius: entity_radius = entity_radius

        self._collider: CircleCollider = CircleCollider(
            position=position,
            radius=collider_radius
        )

        self._color: Tuple[int, int, int] = color
        self._render_collision_outline: bool = render_collision_outline

    def is_colliding(self, other: 'Entity') -> bool:
        if self._id == other._id:
            return False
        
        if self._collider.is_colliding(other._collider):
            return True
        
        return False
    
    def distance_to(self, other: 'Entity') -> float:
        return self._position.distance_to(other._position) - self._radius / 2 - other._radius / 2


    def render(self, screen: pygame.Surface):
        if self._render_collision_outline:
            self._collider.render(screen)
        
        pygame.draw.circle(screen, self._color, self._position, self._radius)