from typing import Tuple, Optional

from utils import Vector2D


class BaseCollider:
    """
    Base class for colliders used in a collision system.
    """

    _COLOR_BASE: Tuple[int, int, int] = (0, 255, 0)
    _COLOR_COLLISION: Tuple[int, int, int] = (255, 0, 0)
    _OUTLINE_WIDTH: int = 1

    def __init__(self, position: Vector2D) -> None:
        """
        Initialize the BaseCollider with a position.

        Args:
            position (Vector2D): The initial position of the collider.
        """
        self._color: Tuple[int, int, int] = self._COLOR_BASE
        self._position: Vector2D = position

    def __str__(self) -> str:
        """
        Return a string representation of the BaseCollider.

        Returns:
            str: A string showing the position of the collider.
        """
        return f"ColliderBase position: ({self._position.x}, {self._position.y})"

    def on_collision_start(self) -> None:
        """
        Change the color of the collider to indicate a collision has started.
        """
        self._color = self._COLOR_COLLISION

    def on_collision_end(self) -> None:
        """
        Change the color of the collider back to the base color to indicate the collision has ended.
        """
        self._color = self._COLOR_BASE

    def set_position(self, new_position: Optional[Vector2D] = None, new_x: Optional[float] = None, new_y: Optional[float] = None) -> None:
        """
        Set the position of the collider.

        You can either provide a Vector2D instance or x and y coordinates.

        Args:
            new_position (Optional[Vector2D]): The new position as a Vector2D instance.
            new_x (Optional[float]): The new x-coordinate.
            new_y (Optional[float]): The new y-coordinate.

        Raises:
            ValueError: If neither new_position nor both new_x and new_y are provided.
        """
        if new_position is not None:
            self._position = new_position
        elif new_x is not None and new_y is not None:
            self._position.x = new_x
            self._position.y = new_y
        else:
            raise ValueError("You must provide either a new_position or both new_x and new_y.")
        
    def get_position(self) -> Vector2D:
        """
        Returns the current position of the collider.

        Returns:
            Vector2D: The current position of the collider as a Vector2D object.
        """

        return self._position
