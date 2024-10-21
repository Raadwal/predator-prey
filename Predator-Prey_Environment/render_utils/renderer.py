import numpy as np
import pygame

from entities import Plant, Prey, Water, Predator

class Renderer:
    _BACKGROUND_COLOR = (34, 139, 34)

    def __init__(
            self,
            render_mode: str,
            window_width: int,
            window_height: int,

            show_fps: bool = True,
    ):
        # Window settings
        self._render_mode: str = render_mode
        self._window_width: int = window_width
        self._window_height: int = window_height
    
        self._show_fps: bool = show_fps

        self._screen: pygame.Surface = None
        self._clock: pygame.time.Clock = None
        self._font: pygame.font.Font = None
        self._is_window_open: bool = False
        self.__initialize_pygame()

        self._water_sources_to_render = []
        self._plants_to_render = []
        self._predators_to_render = []
        self._preys_to_render = []

    def set_pipeline(
            self, 
            water_sources: list[Water],
            plants: list[Plant],
            preys: list[Prey],
            predators: list[Predator],
        ):

        self._plants_to_render = plants
        self._preys_to_render = preys
        self._water_sources_to_render = water_sources
        self._predators_to_render = predators

    def render(self):
        if self._render_mode == 'human':
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return False

            fps = self._clock.get_fps()
            self.__render_entities()

            if self._show_fps:
                fps_text = self._font.render(f"FPS: {int(fps)}", True, (0, 0, 0))
                self._screen.blit(fps_text, (10, 10))

            pygame.display.flip()
            self._clock.tick(30)
            return True

        elif self._render_mode == 'rgb_array':
            self.__render_entities()
            rgb_array = pygame.surfarray.array3d(self._screen)
            return np.transpose(rgb_array, (1, 0, 2))
    
    def close(self):
        if self._is_window_open:
            pygame.quit()
            pygame.font.quit()
            self._is_window_open = False
    
    def __initialize_pygame(self):
        if self._render_mode == 'human':
            pygame.init()
            pygame.font.init()
            pygame.display.set_caption('Predator-Prey Environment')
            self._screen = pygame.display.set_mode((self._window_width, self._window_height))
            self._is_window_open = True   
            self._clock = pygame.time.Clock()
            self._font = pygame.font.SysFont('Arial', 18, bold=True)

        elif self._render_mode == 'rgb_array':
            self._screen = pygame.Surface((self._window_width, self._window_height))

    def __render_entities(self):
        self._screen.fill(self._BACKGROUND_COLOR)

        for water_source in self._water_sources_to_render:
            water_source.render(self._screen)

        for plant in self._plants_to_render:
            plant.render(self._screen)

        for prey in self._preys_to_render:
            prey.render(self._screen)

        for predator in self._predators_to_render:
            predator.render(self._screen)