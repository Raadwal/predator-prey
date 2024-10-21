import pygame
from typing import Tuple
from utils import Vector2D
from .base_collider import BaseCollider

class CircleCollider(BaseCollider):
    """
    CircleCollider does not support rotation.

    Attributes:
        position (Vector2D): The position of the circle collider.
        radius (float): The radius of the circle collider.
    """

    def __init__(self, position: Vector2D, radius: float) -> None:
        """
        Initialize the CircleCollider with a position and a radius.

        Args:
            position (Vector2D): The initial position of the circle collider.
            radius (float): The radius of the circle collider.
        """
        super().__init__(position)
        self._radius: float = radius

    def render(self, screen: pygame.Surface) -> None:
        """
        Render the circle collider on the given screen.

        Args:
            screen (pygame.Surface): The Pygame surface to draw the circle on.
        """
        pygame.draw.circle(screen, self._color, (int(self._position.x), int(self._position.y)), int(self._radius), self._OUTLINE_WIDTH)

    def is_colliding(self, other_circle: 'CircleCollider') -> bool:
        """
        Check if this circle collider is colliding with another circle collider.

        Args:
            other_circle (CircleCollider): The other circle collider to check collision with.

        Returns:
            bool: True if the circles are colliding, False otherwise.
        """
        # Calculate the squared distance between the centers
        distance_x: float = other_circle._position.x - self._position.x
        distance_y: float = other_circle._position.y - self._position.y

        distance_squared: float = distance_x * distance_x + distance_y * distance_y

        # Calculate the squared sum of the radii
        radius_sum: float = self._radius + other_circle._radius
        radius_sum_squared: float = radius_sum * radius_sum

        # Check for collision
        return distance_squared <= radius_sum_squared
    
    def is_colliding_with_margin(self, other_circle: 'CircleCollider', delta: float) -> Tuple[bool, bool]:
        """
        Check if this circle collider is colliding with another circle collider,
        and if it's colliding within an extended range defined by delta.

        Args:
            other_circle (CircleCollider): The other circle collider to check collision with.
            delta (float): The additional margin for collision detection.

        Returns:
            tuple: (bool, bool) A tuple containing two boolean values:
                - True if the circles are colliding without the delta, False otherwise.
                - True if the circles are colliding with the delta, False otherwise.
        """
        # Calculate the squared distance between the centers
        distance_x: float = other_circle._position.x - self._position.x
        distance_y: float = other_circle._position.y - self._position.y

        distance_squared: float = distance_x * distance_x + distance_y * distance_y

        # Calculate the squared sum of the radii
        radius_sum: float = self._radius + other_circle._radius
        radius_sum_squared: float = radius_sum * radius_sum

        # Calculate the squared sum of the radii plus the delta
        radius_sum_with_delta: float = self._radius + other_circle._radius + delta
        radius_sum_with_delta_squared: float = radius_sum_with_delta * radius_sum_with_delta

        # Check for collision without delta and with delta
        collision_without_delta: bool = distance_squared <= radius_sum_squared
        collision_with_delta: bool = distance_squared <= radius_sum_with_delta_squared

        return collision_without_delta, collision_with_delta
    
    def get_radius(self) -> float:
        """
        Returns the radius of the collider.

        Returns:
            float: The radius of the collider.
        """
        return self._radius
