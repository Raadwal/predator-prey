import numpy as np

from typing import Union, Dict, TypedDict

from entities import Water, Plant, Entity, Prey, Predator
from render_utils import Renderer
from utils import Vector2D

class Environment:
    class AnimalInfo(TypedDict):
        drink_count: int
        eat_count: int
        reproduction_count: int
        survived_time_steps: int

    def __init__(
            self,
            env_width: int = 1360,
            env_height: int = 720,
            render_mode: str = 'rgb_array',

            preys_initial_count: int = 50,
            prey_size: int = 15,
            prey_view_distance: Union[int, float] = 150,
            prey_fov: Union[int, float] = 225,
            prey_reproduction: bool = True,
            prey_thirst_decay: bool = True,
            prey_hunger_decay: bool = True,

            plants_count: int = 100,
            plant_size: int = 10,
            plant_growth_time_steps: int = 200,
            plant_random_initialization: bool = True,

            water_sources_count: int = 10,
            water_source_size: int = 75,

            predators_initial_count: int = 1,
            predator_size: int = 15,
            predator_view_distance: Union[int, float] = 300,
            predator_fov: Union[int, float] = 90,
            predator_reproduction: bool = True,
            predator_thirst_decay: bool = True,
            predator_hunger_decay: bool = True,

            collision_detection_scale: float = 0.8,
            enable_precise_collision: bool = True,
            prey_render_collision_outline: bool = False,
            plant_render_collision_outline: bool = False,
            water_source_render_collision_outline: bool = False,
            predator_render_collision_outline: bool = False,
            show_fps: bool = False,
            ):
        
        # Environment settings
        self._env_width: int = env_width
        self._env_height: int = env_height
        self._render_mode: str = render_mode

        # Plant settings
        self._plants_count: int = plants_count
        self._plant_size: int = plant_size
        self._plant_growth_time_steps: int = plant_growth_time_steps
        self._plant_random_initialization: bool = plant_random_initialization

        # Water settings:
        self._water_sources_count: int = water_sources_count
        self._water_source_size: int = water_source_size

        # Prey settings
        self._preys_initial_count: int = preys_initial_count
        self._prey_size: int = prey_size
        self._prey_view_distance: Union[int, float] = prey_view_distance
        self._prey_fov: Union[int, float] = prey_fov
        self._prey_reproduction: bool = prey_reproduction
        self._prey_is_hunger_decay: bool = prey_hunger_decay
        self._prey_is_thirst_decay: bool = prey_thirst_decay
    
        # Predator settings
        self._predators_initial_count: int = predators_initial_count
        self._predator_size: int = predator_size
        self._predator_view_distance: Union[int, float] = predator_view_distance
        self._predator_fov: Union[int, float] = predator_fov
        self._predator_reproduction: bool = predator_reproduction
        self._predator_is_hunger_decay: bool = predator_hunger_decay
        self._predator_is_thirst_decay: bool = predator_thirst_decay

        # Collisions settings
        self._collision_detection_scale: float = collision_detection_scale
        self._enable_precise_collision: bool = enable_precise_collision

        # Render settings
        self._plant_render_collision_outline: bool = plant_render_collision_outline,
        self._prey_render_collision_outline: bool = prey_render_collision_outline
        self._water_source_render_collision_outline: bool = water_source_render_collision_outline
        self._predator_render_collision_outline: bool = predator_render_collision_outline

        # Initialize renderer
        self._renderer = Renderer(
            render_mode=self._render_mode,
            window_width=self._env_width,
            window_height=self._env_height,
            show_fps=show_fps
        )

        self.__initialize_environment()

    def reset(self) -> None:
        self.__initialize_environment()

    def step(self) -> list[Plant]:
        for plant in self._plants:
            plant.step()

    def obs_prey(self, prey: Prey):
        observation = prey.get_observation(
            env_width=self._env_width,
            env_height=self._env_height,
            plants=self._plants,
            water_sources=self._water,
            predators=self._predators
        )

        return observation
    
    def obs_predator(self, predator: Predator):
        observation = predator.get_observation(
            env_width=self._env_width,
            env_height=self._env_height,
            preys=self._preys,
            water_sources=self._water
        )

        return observation
    
    def dummy_step_prey(self, prey: Prey, action: int):
        prey.step(action)

        if not prey.is_alive():
            idx_to_delete = -1
            for idx, other_prey in enumerate(self._preys):
                if other_prey._id == prey._id:
                    idx_to_delete = idx

            if idx_to_delete != -1:
                self._prey_statistics[prey._id] = {
                    "drink_count": prey._drink_count,
                    "eat_count": prey._eat_count,
                    "reproduction_count": prey._reproduction_count,
                    "survived_time_steps": prey._time_steps_survived
                }
                
                del self._preys[idx_to_delete]

        for water_source in self._water:
            collision = prey.is_colliding(water_source)
            if collision:
                prey.drink()

        for plant in self._plants:
            collision = prey.is_colliding(plant)
            if collision:
                prey.eat(plant)


    def step_prey(self, prey: Prey, action: int):
        prey.step(action)
        
        if not prey.is_alive():
            idx_to_delete = -1
            for idx, other_prey in enumerate(self._preys):
                if other_prey._id == prey._id:
                    idx_to_delete = idx
            if idx_to_delete != -1:
                self._prey_statistics[prey._id] = {
                    "drink_count": prey._drink_count,
                    "eat_count": prey._eat_count,
                    "reproduction_count": prey._reproduction_count,
                    "survived_time_steps": prey._time_steps_survived
                }

                del self._preys[idx_to_delete]

        if prey.is_reproducing():
            for _ in range(2):
                self.reproduce_prey(prey)

        for water_source in self._water:
            collision = prey.is_colliding(water_source)
            if collision:
                prey.drink()

        for plant in self._plants:
            collision = prey.is_colliding(plant)
            if collision:
                prey.eat(plant)

        observation = prey.get_observation(
            env_width=self._env_width,
            env_height=self._env_height,
            plants=self._plants,
            water_sources=self._water,
            predators=self._predators
        )

        reward = prey.get_reward(self._env_width, self._env_height)
        done = not prey.is_alive()

        return observation, reward, done
    
    def step_predator(self, predator: Predator, action: int):
        predator.step(action)

        if not predator.is_alive():
            idx_to_delete = -1
            for idx, other_predator in enumerate(self._predators):
                if other_predator._id == predator._id:
                    idx_to_delete = idx
            if idx_to_delete != -1:
                self._predator_statistics[predator._id] = {
                    "drink_count": predator._drink_count,
                    "eat_count": predator._eat_count,
                    "reproduction_count": predator._reproduction_count,
                    "survived_time_steps": predator._time_steps_survived
                }
                del self._predators[idx_to_delete]

        if predator.is_reproducing():
            self.reproduce_predator(predator)

        for water_source in self._water:
            collision = predator.is_colliding(water_source)
            if collision:
                predator.drink()

        for prey in self._preys:
            collision = predator.is_colliding(prey)
            if collision:
                predator.eat(prey)

        observation = predator.get_observation(
            env_width=self._env_width,
            env_height=self._env_height,
            preys=self._preys,
            water_sources=self._water
        )

        reward = predator.get_reward(self._env_width, self._env_height)
        done = not predator.is_alive()

        return observation, reward, done

    def get_preys(self) -> list[Prey]:
        return self._preys
    
    def get_predators(self) -> list[Predator]:
        return self._predators

    def render(self):
        self._renderer.set_pipeline(
            water_sources=self._water,
            plants=self._plants,
            preys=self._preys,
            predators=self._predators,
        )

        return self._renderer.render()
    
    def close(self):
        self._renderer.close()

    def __initialize_environment(self) -> None:
        self._entity_new_id: int = 0

        self._water = []
        self.__initialize_water()

        self._plants = []
        self.__initialize_plants()

        self._preys = []
        self.__initialize_preys()

        self._predators = []
        self.__initialize_predators()

        self._prey_statistics: Dict[int, Environment.AnimalInfo] = {}
        self._predator_statistics: Dict[int, Environment.AnimalInfo] = {}

    def __is_colliding(self, entity: Entity, other_entities: Entity) -> bool:
        for other_entity in other_entities:
            if entity.is_colliding(other_entity):
                return True
            
        return False
    
    def __initialize_water(self):
        i: int = 0

        while i < self._water_sources_count:
            pos_x: int = np.random.randint(self._plant_size // 5, (self._env_width - self._plant_size + 1) // 5) * 5
            pos_y: int = np.random.randint(self._plant_size // 5, (self._env_height - self._plant_size + 1) // 5) * 5


            water: Water = Water (
                id=self._entity_new_id,
                position=Vector2D(pos_x, pos_y),
                entity_radius=self._water_source_size,
                collider_radius=self._water_source_size,
                render_collision_outline=self._water_source_render_collision_outline
            )

            if self.__is_colliding(water, self._water):
                continue
            
            self._water.append(water)
            i+=1
            self._entity_new_id += 1


    def __initialize_plants(self):
        i: int = 0

        while i < self._plants_count:
            pos_x: int = np.random.randint(self._plant_size // 5, (self._env_width - self._plant_size + 1) // 5) * 5
            pos_y: int = np.random.randint(self._plant_size // 5, (self._env_height - self._plant_size + 1) // 5) * 5


            plant: Plant = Plant(
                id=self._entity_new_id,
                position=Vector2D(pos_x, pos_y),
                entity_radius=self._plant_size,
                collider_radius=self._plant_size,
                growth_time_steps=self._plant_growth_time_steps,
                random_initialization=self._plant_random_initialization,
                render_collision_outline=self._plant_render_collision_outline,
            )

            if self.__is_colliding(plant, self._water):
                continue

            if self.__is_colliding(plant, self._plants):
                continue

            self._plants.append(plant)
            i += 1
            self._entity_new_id += 1
    
    def __initialize_preys(self):
        for _ in range(self._preys_initial_count):
            prey = self.__initialize_prey()

            self._preys.append(prey)

    def __initialize_prey(self) -> Prey:
        spawned = False

        while not spawned:
            pos_x = np.random.randint(self._prey_size // 5, (self._env_width - self._prey_size + 1) // 5) * 5
            pos_y = np.random.randint(self._prey_size // 5, (self._env_height - self._prey_size + 1) // 5) * 5

            entity_orientation_angle = np.random.randint(0, 73) * 5

            prey: Prey = Prey(
                id=self._entity_new_id,
                position=Vector2D(pos_x, pos_y),
                entity_orientation_angle=entity_orientation_angle,
                entity_radius=self._prey_size,
                collider_radius=self._prey_size,
                view_distance=self._prey_view_distance,
                fov=self._prey_fov,
                reproduction=self._prey_reproduction,
                hunger_decay=self._prey_is_hunger_decay,
                thirst_decay=self._prey_is_thirst_decay,
                collision_detection_scale=self._collision_detection_scale,
                enable_precise_collision=self._enable_precise_collision,
                render_collision_outline=self._prey_render_collision_outline
            )

            if self.__is_colliding(prey, self._water):
                continue
        
            if self.__is_colliding(prey, self._plants):
                continue

            if self.__is_colliding(prey, self._preys):
                continue

            spawned = True

        self._entity_new_id += 1

        return prey

    def __initialize_predators(self):
        for _ in range(self._predators_initial_count):
            predator = self.__initialize_predator()

            self._predators.append(predator)
    
    def __initialize_predator(self) -> Predator:
        spawned = False

        while not spawned:
            pos_x = np.random.randint(self._predator_size // 5, (self._env_width - self._predator_size + 1) // 5) * 5
            pos_y = np.random.randint(self._predator_size // 5, (self._env_height - self._predator_size + 1) // 5) * 5

            entity_orientation_angle = np.random.randint(0, 73) * 5

            predator: Predator = Predator(
                id=self._entity_new_id,
                position=Vector2D(pos_x, pos_y),
                entity_orientation_angle=entity_orientation_angle,
                entity_radius=self._predator_size,
                collider_radius=self._predator_size,
                view_distance=self._predator_view_distance,
                fov=self._predator_fov,
                reproduction=self._predator_reproduction,
                hunger_decay=self._predator_is_hunger_decay,
                thirst_decay=self._predator_is_thirst_decay,
                collision_detection_scale=self._collision_detection_scale,
                enable_precise_collision=self._enable_precise_collision,
                render_collision_outline=self._predator_render_collision_outline
            )

            if self.__is_colliding(predator, self._water):
                continue
        
            if self.__is_colliding(predator, self._plants):
                continue

            if self.__is_colliding(predator, self._preys):
                continue

            if self.__is_colliding(predator, self._predators):
                continue

            spawned = True

        self._entity_new_id += 1

        return predator

    def reproduce_prey(self, parent: Prey) -> bool:
        additional_distance = 0
        max_distance_distance = 100

        for distance in range(additional_distance, max_distance_distance + 1, 5):
            spawn_distance = 2 * parent._collider._radius + distance

            parent_pos_x = parent._collider._position.x
            parent_pos_y = parent._collider._position.y

            possible_positions = [
                Vector2D(parent_pos_x + spawn_distance, parent_pos_y),
                Vector2D(parent_pos_x - spawn_distance, parent_pos_y),
                Vector2D(parent_pos_x, parent_pos_y+ spawn_distance),
                Vector2D(parent_pos_x, parent_pos_y - spawn_distance),
            ]

            for position in possible_positions:
                if position.x < -200 or position.x > self._env_width + 200:
                    continue
                if position.y < - 200 or position.y > self._env_height + 200:
                    continue

                entity_orientation_angle = np.random.randint(0, 73) * 5

                prey: Prey = Prey(
                    id=self._entity_new_id,
                    position=position,
                    entity_orientation_angle=entity_orientation_angle,
                    entity_radius=self._prey_size,
                    collider_radius=self._prey_size,
                    view_distance=self._prey_view_distance,
                    fov=self._prey_fov,
                    reproduction=self._prey_reproduction,
                    hunger_decay=self._prey_is_hunger_decay,
                    thirst_decay=self._prey_is_thirst_decay,
                    collision_detection_scale=self._collision_detection_scale,
                    enable_precise_collision=self._enable_precise_collision,
                    render_collision_outline=self._prey_render_collision_outline
                )

                if self.__is_colliding(prey, self._water):
                    continue

                if self.__is_colliding(prey, self._preys):
                    continue

                if self.__is_colliding(prey, self._predators):
                    continue

                self._entity_new_id += 1
                parent._time_to_reproduction = parent._reproduction_time
                parent._reproduction_this_time_step = True
                parent._reproduction_count += 1
                self._preys.append(prey)
                return True
            
        parent._reproduction_this_time_step = True
        return False
    
    
    def reproduce_predator(self, parent: Predator) -> bool:
        additional_distance = 0
        max_distance_distance = 100

        for distance in range(additional_distance, max_distance_distance + 1, 5):
            spawn_distance = 2 * parent._collider._radius + distance

            parent_pos_x = parent._collider._position.x
            parent_pos_y = parent._collider._position.y

            possible_positions = [
                Vector2D(parent_pos_x + spawn_distance, parent_pos_y),
                Vector2D(parent_pos_x - spawn_distance, parent_pos_y),
                Vector2D(parent_pos_x, parent_pos_y+ spawn_distance),
                Vector2D(parent_pos_x, parent_pos_y - spawn_distance),
            ]

            for position in possible_positions:
                if position.x < -200 or position.x > self._env_width + 200:
                    continue
                if position.y < - 200 or position.y > self._env_height + 200:
                    continue

                entity_orientation_angle = np.random.randint(0, 73) * 5

                prey: Predator = Predator(
                    id=self._entity_new_id,
                    position=position,
                    entity_orientation_angle=entity_orientation_angle,
                    entity_radius=self._predator_size,
                    collider_radius=self._predator_size,
                    view_distance=self._predator_view_distance,
                    fov=self._predator_fov,
                    reproduction=self._predator_reproduction,
                    hunger_decay=self._predator_is_hunger_decay,
                    thirst_decay=self._predator_is_thirst_decay,
                    collision_detection_scale=self._collision_detection_scale,
                    enable_precise_collision=self._enable_precise_collision,
                    render_collision_outline=self._predator_render_collision_outline
                )

                if self.__is_colliding(prey, self._water):
                    continue

                if self.__is_colliding(prey, self._preys):
                    continue

                self._entity_new_id += 1
                parent._time_to_reproduction = parent._reproduction_time
                parent._reproduction_this_time_step = True
                parent._reproduction_count += 1
                self._predators.append(prey)
                return True

        parent._reproduction_this_time_step = True
        return False
    
    def get_preys_statistics(self) -> Dict[int, AnimalInfo]:
        for prey in self._preys:
            self._prey_statistics[prey._id] = {
                "drink_count": prey._drink_count,
                "eat_count": prey._eat_count,
                "reproduction_count": prey._reproduction_count,
                "survived_time_steps": prey._time_steps_survived
            }

        return self._prey_statistics
    
    def get_predators_statistics(self) -> Dict[int, AnimalInfo]:
        for predator in self._predators:
            self._predator_statistics[predator._id] = {
                "drink_count": predator._drink_count,
                "eat_count": predator._eat_count,
                "reproduction_count": predator._reproduction_count,
                "survived_time_steps": predator._time_steps_survived
            }

        return self._predator_statistics

    def get_plants_statistics(self) -> Dict[int, int]:
        plant_statistics = {}

        for plant in self._plants:
            plant_statistics[plant._id] = plant._been_eaten_count

        return plant_statistics