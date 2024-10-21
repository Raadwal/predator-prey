import pygame
import math

from .entity import Entity
from .plant import Plant
from .water import Water
from typing import Tuple

import heapq

import random
from colliders import CircleCollider, ConeCollider
from utils import Vector2D, scale_value

class Animal(Entity):

    def __init__(
        self, 
        id: int,
        position: Vector2D,
        entity_orientation_angle: float,
        entity_radius: float,
        collider_radius: float,
        color: Tuple[int, int, int],
        view_distance: float,
        fov: float,
        reproduction: bool,
        thirst_decay: bool,
        hunger_decay: bool,
        collision_detection_scale: float,
        enable_precise_collision: bool = False,
        render_collision_outline: bool = False
    ) -> None:

        super().__init__(
            id=id,
            position=position,
            entity_radius=entity_radius,
            collider_radius=collider_radius,
            color=color,
            render_collision_outline=render_collision_outline
        )

        self._entity_orientation_angle = entity_orientation_angle

        self._collider: CircleCollider = CircleCollider(
            position=position,
            radius=collider_radius
        )

        self._view_cone: ConeCollider = ConeCollider(
            position=position,
            view_distance=view_distance,
            fov=fov,
            orientation_angle=entity_orientation_angle,
            collision_detection_scale=collision_detection_scale,
            enable_precise_collision=enable_precise_collision
        )

        self._is_reproduction = reproduction
        self._is_thirst_decay = thirst_decay
        self._is_hunger_decay = hunger_decay
        self._is_alive = True

        self._action_dict = {
            0: lambda: self._move_forward(5),
            1: lambda: self._change_direction(5),
            2: lambda: self._change_direction(-5),
        }

        self._max_entities_in_view = 1

        # Settings
        self._hunger_max = 200
        self._hunger = self._hunger_max
        self._hunger_decay = 0.5
        self._eaten_this_time_step = False

        self._thirst_max = 200
        self._thirst = self._hunger_max
        self._thirst_decay = 0.5
        self._drunk_this_time_step = False

        self._reproduction_time = 200
        self._time_to_reproduction = self._reproduction_time
        self._reproduction_decay = 0.5
        self._reproduction_this_time_step = False

        self._drink_count = 0
        self._eat_count = 0
        self._reproduction_count = 0
        self._time_steps_survived = 0

    def eat(self, entity: Entity):
        raise NotImplementedError("The eat method has not been implemented yet.")

    def drink(self, entity: Entity):
        raise NotImplementedError("The drink method has not been implemented yet.")
    
    def is_reproducing(self) -> bool:
        if not self._is_reproduction:
            return False
        
        if self._time_to_reproduction <= 0:
            return True
        
        return False

    def _move_forward(self, distance):
        rad = math.radians(-self._entity_orientation_angle)
        self._position.x += distance * math.cos(rad)
        self._position.y += distance * math.sin(rad)

    def _change_direction(self, angle_change):
        self._entity_orientation_angle = (self._entity_orientation_angle + angle_change) % 360
        self._view_cone.change_angle(angle_change)

    def step(self, action) -> None:
        if self._reproduction_this_time_step:
            self._reproduction_this_time_step = False

        self._time_steps_survived += 1

        self._action_dict[action]()

        if self._is_hunger_decay:
            self._hunger -= self._hunger_decay
        if self._is_thirst_decay:
            self._thirst -= self._thirst_decay

        self._time_to_reproduction -= self._reproduction_decay

        if self._hunger <= 0 or self._thirst <=0:
            self._is_alive = False
        
    def get_position(self) -> Vector2D:
        return self._position
    
    def set_position(self, x, y) -> None:
        self._position.x = x
        self._position.y = y

    def is_alive(self):
        return self._is_alive

    def get_random_action(self):
        return random.randint(0, len(self._action_dict) - 1)

    def render(self, screen: pygame.Surface):
        if self._render_collision_outline:
            self._collider.render(screen)
            self._view_cone.render(screen)
        
        pygame.draw.circle(screen, self._color, self._position, self._radius)

        end_x = self._position.x + self._radius * math.cos(math.radians(-self._entity_orientation_angle))
        end_y = self._position.y + self._radius * math.sin(math.radians(-self._entity_orientation_angle))

        pygame.draw.line(screen, (255, 255, 255), (int(self._position.x), int(self._position.y)), (int(end_x), int(end_y)), 2)

    
    def _process_water_source(self, closes_water_source, water_source, type: int):
        distance_x = self._collider._position.x - water_source._collider._position.x
        distance_y = self._collider._position.y - water_source._collider._position.y

        distance = math.sqrt(distance_x ** 2 + distance_y ** 2)

        dx = water_source._collider._position.x - self._position.x
        dy = self._position.y - water_source._collider._position.y
        angle = math.degrees(math.atan2(dy, dx))

        heapq.heappush(closes_water_source, (-distance, angle, type, water_source._position.x, water_source._position.y))
        if len(closes_water_source) > 1:
                heapq.heappop(closes_water_source)


    def _process_entity(self, closest_entities: list, entity: Entity, type: int):
        if self._id == entity._id:
            return
        
        collides, distance, angle = self._view_cone.is_colliding(entity._collider)

        if collides:
            heapq.heappush(closest_entities, (-distance, angle, type, entity._position.x, entity._position.y))
            if len(closest_entities) > self._max_entities_in_view:
                heapq.heappop(closest_entities)


    def get_reward(self):
        raise NotImplementedError("The get_reward method has not been implemented yet.")