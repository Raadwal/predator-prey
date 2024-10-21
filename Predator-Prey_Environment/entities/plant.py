import pygame

import random
from typing import Tuple

from utils import Vector2D
from colliders import CircleCollider
from .entity import Entity

class Plant(Entity):
    COLOR: Tuple[int, int, int] = (0, 100, 0)
    EATEN_COLOR: Tuple[int, int, int] = (139, 69, 19)

    def __init__(
        self, 
        id: int,
        position: Vector2D,
        entity_radius: float,
        collider_radius: float,
        growth_time_steps: int,
        random_initialization: bool = True,
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
        
        # Plant settings
        self._eaten_color: Tuple[int, int, int] = self.EATEN_COLOR
        self._growth_time_steps: int = growth_time_steps
        self._current_time_step: int = 0
        self._is_grown = False

        if random_initialization:
            self._is_grown: bool = random.choice([True, False])
        
            if not self._is_grown:
                self._current_time_step = random.randint(0, self._growth_time_steps)

        self._been_eaten_count = 0
        
    def step(self):
        if self._is_grown:
            return
        
        self._current_time_step += 1

        if self._current_time_step >= self._growth_time_steps:
            self._is_grown = True

    def eat(self):
        if self._is_grown:
            self._current_time_step = 0
            self._is_grown = False
            self._been_eaten_count += 1
            return True

        return False

    def render(self, screen: pygame.Surface):
        if self._render_collision_outline:
            self._collider.render(screen)
        
        if self._is_grown:
            pygame.draw.circle(screen, self._color, self._position, self._radius)
        else:
            growth_percentage: float = self._current_time_step / self._growth_time_steps
            growth_radius: float = self._radius * growth_percentage
            pygame.draw.circle(screen, self._eaten_color, self._position, self._radius)
            pygame.draw.circle(screen, self._color, self._position, growth_radius)
        