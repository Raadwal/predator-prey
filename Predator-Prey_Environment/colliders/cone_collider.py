import pygame

import math

from typing import Union, Tuple

from utils import Vector2D
from .base_collider import BaseCollider
from .circle_collider import CircleCollider

class ConeCollider(BaseCollider):
    """
    A class to represent a cone-shaped collider for collision detection.

    Attributes:
        position (Vector2D): The position of the cone collider.
        view_distance (float or int): The maximum distance the cone collider can detect collisions.
        fov (float or int): The field of view angle of the cone collider in degrees.
        orientation_angle (float or int): The angle at which the cone collider is oriented.
        collision_detection_scale (float): The scale factor for collision detection sensitivity, ranging from 0 to 1.
        enable_precise_collision (bool): Indicates whether to use precise but slower collision detection or faster but less precise detection.

    Methods:
        __init__(position, view_distance, fov, orientation_angle, collision_detection_scale, enable_precise_collision):
            Initializes the ConeCollider with the specified attributes.
    """

    def __init__(
            self, 
            position: Vector2D,
            view_distance: Union[float, int], 
            fov: Union[float, int],
            orientation_angle: Union[float, int],
            collision_detection_scale: float,
            enable_precise_collision: bool
            ) -> None:
        """
        Initializes the ConeCollider with the specified attributes.

        Args:
            position (Vector2D): The position of the cone collider.
            view_distance (float or int): The maximum distance the cone collider can detect collisions.
            fov (float or int): The field of view angle of the cone collider in degrees.
            orientation_angle (float or int): The angle at which the cone collider is oriented.
            collision_detection_scale (float): The scale factor for collision detection sensitivity, ranging from 0 to 1.
            enable_precise_collision (bool): Indicates whether to use precise but slower collision detection or faster but less precise detection.
        
        Raises:
            ValueError: If fov is not between 0 and 360.
            ValueError: If orientation_angle is not between 0 and 360.
            ValueError: If collision_detection_scale is not between 0 and 1.
        """
        if not (0 <= fov <= 360):
            raise ValueError("fov must be between 0 and 360 degrees.")
        if not (0 <= orientation_angle <= 360):
            raise ValueError("orientation_angle must be between 0 and 360 degrees.")
        if not (0 <= collision_detection_scale <= 1):
            raise ValueError("collision_detection_scale must be between 0 and 1.")

        super().__init__(position)
        self._view_distance: Union[float, int] = view_distance
        self._fov: Union[float, int] = fov
        self._orientation_angle: Union[float, int] = orientation_angle
        self._collision_detection_scale: float = collision_detection_scale
        self._enable_precise_collision: float = enable_precise_collision

    def render(self, screen: pygame.Surface) -> None:
        """
        Renders the cone collider on the given Pygame screen.

        Args:
            screen (pygame.Surface): The Pygame surface to draw the cone collider on.
        """
        half_fov_radians: float = math.radians(self._fov / 2)
        orientation_angle_radians: float = math.radians(self._orientation_angle)

        # Drawing arc - no angle adjustment for pygame needed
        start_angle: float =  orientation_angle_radians  - half_fov_radians
        end_angle: float = orientation_angle_radians + half_fov_radians

        view_distance_doubled: Union[float, int] = 2 * self._view_distance
        rect: pygame.Rect = pygame.Rect(
            self._position.x - self._view_distance, 
            self._position.y - self._view_distance, 
            view_distance_doubled, view_distance_doubled
        )
        pygame.draw.arc(screen, self._color, rect, start_angle, end_angle, self._OUTLINE_WIDTH)

        # Skip drawing lines if the field of view is 360 degrees
        if self._fov == 360:
            return

        # Drawing lines - angle adjustment for pygame coordinates system
        start_angle = -orientation_angle_radians - half_fov_radians
        end_angle = -orientation_angle_radians + half_fov_radians

        start_angle_point: Vector2D = self._get_point(self._position, self._view_distance, start_angle)
        end_angle_point: Vector2D = self._get_point(self._position, self._view_distance, end_angle)

        pygame.draw.line(screen, self._color, self._position.to_tuple(), start_angle_point.to_tuple(), self._OUTLINE_WIDTH)
        pygame.draw.line(screen, self._color, self._position.to_tuple(), end_angle_point.to_tuple(), self._OUTLINE_WIDTH)

    def _get_point(self, start_point: Vector2D, distance: float, angle: float) -> Vector2D:
        """
        Calculates the endpoint of a line given a starting point, distance, and angle.

        Args:
            start_point (Vector2D): The starting point of the line.
            distance (float): The length of the line.
            angle (float): The angle in radians at which the line is drawn.

        Returns:
            Vector2D: The endpoint of the line.
        """
        return Vector2D(
            start_point.x + distance * math.cos(angle), 
            start_point.y + distance * math.sin(angle)
        )

    def is_colliding(self, circle_collider: 'CircleCollider') -> Tuple[bool, float, float]:
        """
        Determines if the cone collider is colliding with a given circle collider and returns the distance and angle.

        Args:
            circle_collider (CircleCollider): The circle collider to check for a collision.

        Returns:
            tuple: (bool, float, float) A tuple containing:
                - True if the cone collider is colliding with the circle collider, False otherwise.
                - The distance to the circle collider.
                - The angle to the circle collider.
                If not colliding, returns (False, 0, 0).
        """
        # Adjust collision detection based on the collision detection scale
        object_in_cone: float = (1 - self._collision_detection_scale) * circle_collider.get_radius()

        # Calculate the distance between the cone's position and the circle's position
        distance_x: float = circle_collider.get_position().x - self._position.x
        distance_y: float = circle_collider.get_position().y - self._position.y
        distance: float = math.sqrt(distance_x * distance_x + distance_y * distance_y)

        # Calculate the effective radius sum for collision detection
        radius_sum: float = self._view_distance - object_in_cone + circle_collider.get_radius()

        # Check if the circle is outside the cone's effective radius
        if distance > radius_sum:
            return (False, 0, 0)

        # Calculate the angle between the cone's orientation and the line connecting the cone to the circle
        distance_vector: Vector2D = Vector2D(circle_collider.get_position().x - self._position.x, circle_collider.get_position().y - self._position.y)
        angle_to_cone: float = math.degrees(-math.atan2(distance_vector.y, distance_vector.x))
        angle_diff: float = abs((angle_to_cone - self._orientation_angle + 180) % 360 - 180)

        # Check if the angle difference is within the cone's field of view
        if angle_diff <= self._fov / 2:
            return (True, distance, angle_to_cone)

        # If precise collision detection is not enabled, return False
        if not self._enable_precise_collision:
            return (False, 0, 0)

        # Calculate the start and end angles of the cone in radians
        cone_angle: float = math.radians(-self._orientation_angle)
        cone_angle_span: float = math.radians(self._fov)
        start_angle: float = cone_angle - cone_angle_span / 2
        end_angle: float = cone_angle + cone_angle_span / 2

        # Calculate the endpoints of the cone's boundary lines
        start_point: Vector2D = self._get_point(self._position, self._view_distance, start_angle)
        end_point: Vector2D = self._get_point(self._position, self._view_distance, end_angle)

        # Calculate the shortest distances from the circle's position to the cone's boundary lines
        shortest_distance_start: float = self._shortest_distance(circle_collider.get_position(), self._position, start_point)
        shortest_distance_end: float = self._shortest_distance(circle_collider.get_position(), self._position, end_point)

        # Adjust the shortest distances based on the collision detection scale
        shortest_distance_start += object_in_cone
        shortest_distance_end += object_in_cone

        # Check if the adjusted shortest distances are less than the circle's radius
        if shortest_distance_start < circle_collider.get_radius() or shortest_distance_end < circle_collider.get_radius():
            return (True, distance, angle_to_cone)

        return (False, 0, 0)

    def _shortest_distance(self, point: Vector2D, line_start: Vector2D, line_end: Vector2D) -> float:
        """
        Calculates the shortest distance from a point to a line segment.

        Args:
            point (Vector2D): The point from which the distance is calculated.
            line_start (Vector2D): The starting point of the line segment.
            line_end (Vector2D): The ending point of the line segment.

        Returns:
            float: The shortest distance from the point to the line segment.
        """

        # Vector from line_start to point and line_start to line_end
        point_to_line_start_x: float = point.x - line_start.x
        point_to_line_start_y: float = point.y - line_start.y
        line_start_to_line_end_x: float = line_end.x - line_start.x
        line_start_to_line_end_y: float = line_end.y - line_start.y

        # Length squared of the line segment
        line_segment_length_squared: float = (
            line_start_to_line_end_x * line_start_to_line_end_x +
            line_start_to_line_end_y * line_start_to_line_end_y
        )

        # Avoid division by zero if the line_start and line_end are the same point
        if line_segment_length_squared == 0:
            return math.sqrt(
                point_to_line_start_x * point_to_line_start_x + 
                point_to_line_start_y * point_to_line_start_y
            )

        # Projection factor
        projection_factor: float = (
            (point_to_line_start_x * line_start_to_line_end_x + 
             point_to_line_start_y * line_start_to_line_end_y) / 
            line_segment_length_squared
        )

        # Ensure projection factor is within the range [0, 1]
        projection_factor = max(0, min(1, projection_factor))

        # Closest point on the line segment to the point
        closest_point_on_line_x: float = line_start.x + projection_factor * line_start_to_line_end_x
        closest_point_on_line_y: float = line_start.y + projection_factor * line_start_to_line_end_y

        # Distance from the point to the closest point on the line segment
        distance_x: float = point.x - closest_point_on_line_x
        distance_y: float = point.y - closest_point_on_line_y

        return math.sqrt(distance_x * distance_x + distance_y * distance_y)
    
    def change_angle(self, angle_change):
        self._orientation_angle = (self._orientation_angle + angle_change) % 360

    def get_orientation_angle(self):
        return self._orientation_angle