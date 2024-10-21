from .animal import Animal
from .prey import Prey
from .entity import Entity
from typing import Tuple

import math
import numpy as np
from utils import Vector2D, scale_value

class Predator(Animal):
    COLOR: Tuple[int, int, int] = (255, 89, 0)

    def __init__(
        self, 
        id: int,
        position: Vector2D,
        entity_orientation_angle: float,
        entity_radius: float,
        collider_radius: float,
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
            entity_orientation_angle=entity_orientation_angle,
            entity_radius=entity_radius,
            collider_radius=collider_radius,
            color=self.COLOR,
            view_distance=view_distance,
            fov=fov,
            reproduction=reproduction,
            thirst_decay=thirst_decay,
            hunger_decay=hunger_decay,
            collision_detection_scale=collision_detection_scale,
            enable_precise_collision=enable_precise_collision,
            render_collision_outline=render_collision_outline
        )

        self._action_dict = {
            0: lambda: self._move_forward(5),
            1: lambda: self._move_forward_fast(10),
            2: lambda: self._change_direction(5),
            3: lambda: self._change_direction(-5),
        }

        self._hunger_max = 200
        self._hunger = self._hunger_max

        self._reproduction_time = 300
        self._time_to_reproduction = self._reproduction_time

    def eat(self, prey: Prey):
        prey._is_alive = False

        self._hunger = self._hunger_max
        self._eat_count += 1

        self._eaten_this_time_step = True

    def drink(self):
        self._thirst = self._thirst_max
        self._drink_count += 1

        self._drunk_this_time_step = True
        self._change_direction(-180)
        self._move_forward(10)

    def get_reward(self, env_width: int, env_height: int) -> float:
        reward = 0

        if self._position.x < 0 or self._position.y < 0 or self._position.x > env_width or self._position.y > env_height:
            reward -= 0.2

        if self._eaten_this_time_step:
            self._eaten_this_time_step = False
            self._eat_count += 1
            reward += 5

        if self._drunk_this_time_step:
            self._drunk_this_time_step = False
            self._drink_count += 1
            reward += 5

        if self._hunger >= 100:
            reward += 0.1

        if self._thirst >= 50:
            reward += 0.1

        if not self._is_alive:
            reward -= 100
        
        return reward
    
    def get_observation(
            self,
            env_width: int,
            env_height: int,
            preys: list[Entity],
            water_sources: list[Entity]
            ):
        
        closest_water_source = []

        for water_source in water_sources:
            self._process_water_source(closest_water_source, water_source, 0)

        water_source_array = np.zeros(6)
        for i, (distance, angle, entity_type, pos_x, pos_y) in enumerate(closest_water_source):
            scaled_distance = scale_value(-distance, 0, math.sqrt(env_width**2 + env_height**2), 0, 1)
            scaled_angle = scale_value(angle, -180, 180, -1, 1)
            scaled_pos_x = scale_value(pos_x, 0, env_width, 0, 1)
            scaled_pos_y = scale_value(pos_y, 0, env_height, 0, 1)
            dx = pos_x - self._position.x
            dy = self._position.y - pos_y
            angle_to_enemy = math.degrees(math.atan2(dy, dx))
            angle_to_enemy  = (angle_to_enemy + 360) % 360
            difference = angle_to_enemy - self._entity_orientation_angle

            if difference > 180:
                difference -= 360
            elif difference < -180:
                difference += 360

            scaled_difference = scale_value(difference, -180, 180, -1, 1)

            water_source_array[i * 6:(i * 6) + 6] = [entity_type, scaled_distance, scaled_angle, scaled_difference, scaled_pos_x, scaled_pos_y]

        closest_entities = []
        for prey in preys:
            self._process_entity(closest_entities, prey, 1)

        result_array = np.zeros(6 * self._max_entities_in_view)

        for i, (distance, angle, entity_type, pos_x, pos_y) in enumerate(closest_entities):
            scaled_distance = scale_value(-distance, 0, self._view_cone._view_distance, 0, 1)
            scaled_angle = scale_value(angle, -180, 180, -1, 1)
            scaled_pos_x = scale_value(pos_x, 0, env_width, 0, 1)
            scaled_pos_y = scale_value(pos_y, 0, env_height, 0, 1)
            dx = pos_x - self._position.x
            dy = self._position.y - pos_y
            angle_to_enemy = math.degrees(math.atan2(dy, dx))
            angle_to_enemy  = (angle_to_enemy + 360) % 360
            difference = angle_to_enemy - self._entity_orientation_angle

            if difference > 180:
                difference -= 360
            elif difference < -180:
                difference += 360

            scaled_difference = scale_value(difference, -self._view_cone._fov/2, self._view_cone._fov/2, -1, 1)

            result_array[i * 6:(i * 6) + 6] = [entity_type, scaled_distance, scaled_angle, scaled_difference, scaled_pos_x, scaled_pos_y]

        scaled_hunger = scale_value(self._hunger, 0, self._hunger_max, 0, 1)
        scaled_thirst = scale_value(self._thirst, 0, self._thirst_max, 0, 1)
        scaled_x = scale_value(self._position.x, 0, env_width, 0, 1)
        scaled_y = scale_value(self._position.y, 0, env_height, 0, 1)
        scaled_orientation_angle = scale_value(self._entity_orientation_angle, 0, 360, 0, 1)
        values_to_add = np.array([scaled_hunger, scaled_thirst, scaled_x, scaled_y, scaled_orientation_angle])

        result_array = np.insert(result_array, 0, values_to_add)
        result_array = np.append(result_array, water_source_array)

        return result_array

    def _move_forward(self, distance):
        rad = math.radians(-self._entity_orientation_angle)
        self._position.x += distance * math.cos(rad)
        self._position.y += distance * math.sin(rad)

    def _move_forward_fast(self, distance):
        rad = math.radians(-self._entity_orientation_angle)
        self._position.x += distance * math.cos(rad)
        self._position.y += distance * math.sin(rad)
        if self._is_hunger_decay:
            self._hunger -= 5