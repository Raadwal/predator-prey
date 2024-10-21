import math
from typing import Union, Iterator, Tuple
from dataclasses import dataclass

@dataclass
class Vector2D:
    """
    A class to represent a two-dimensional vector.

    Attributes:
        x (float): The x-coordinate of the vector.
        y (float): The y-coordinate of the vector.
    """
    x: float = 0.0
    y: float = 0.0

    def rotate(self, angle_degrees: Union[int, float], origin: "Vector2D") -> None:
        """
        Rotate the vector around a given origin by a specified angle in degrees.

        Args:
            angle_degrees (Union[int, float]): The angle in degrees to rotate the vector.
            origin (Vector2D): The origin point around which to rotate the vector.

        Returns:
            None: This method modifies the vector in place.
        """
        angle_radians: float = math.radians(angle_degrees)

        # Translate point back to origin
        temp_x: Union[int, float] = self.x - origin.x
        temp_y: Union[int, float] = self.y - origin.y

        rotated_x: float = temp_x * math.cos(angle_radians) - temp_y * math.sin(angle_radians)
        rotated_y: float = temp_x * math.sin(angle_radians) + temp_y * math.cos(angle_radians)

        # Translate point back
        self.x = rotated_x + origin.x
        self.y = rotated_y + origin.y

    def distance_to(self, other: "Vector2D") -> float:
        """
        Calculate the Euclidean distance between this vector and another vector.

        Args:
            other (Vector2D): The other vector to calculate the distance to.

        Returns:
            float: The Euclidean distance between the two vectors.
        """
        distance = math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
        return distance

    def to_tuple(self) -> Tuple[float, float]:
        """
        Convert the Vector2D to a tuple.

        Returns:
            Tuple[float, float]: A tuple representation of the vector.
        """
        return (self.x, self.y)
    
    @classmethod
    def from_tuple(cls, t: Tuple[float, float]) -> "Vector2D":
        """
        Create a Vector2D instance from a tuple.

        Args:
            t (Tuple[float, float]): A tuple containing the x and y coordinates.

        Returns:
            Vector2D: The created Vector2D instance.
        """
        if len(t) != 2:
            raise ValueError("Tuple must have exactly two elements")
        
        return cls(t[0], t[1])

    def __str__(self) -> str:
        """
        Return a string representation of the Vector2D.

        Returns:
            str: The string representation of the vector.
        """
        return f"Vector2D(x={self.x}, y={self.y})"
    
    def __repr__(self) -> str:
        """
        Return a detailed string representation of the Vector2D.

        Returns:
            str: The detailed string representation of the vector.
        """
        return f"Vector2D(x={self.x}, y={self.y})"

    def __iter__(self) -> Iterator[float]:
        """
        Return an iterator over the vector's coordinates.

        Returns:
            Iterator[float]: An iterator over the vector's coordinates.
        """
        return iter((self.x, self.y))

    def __len__(self) -> int:
        """
        Return the number of dimensions of the vector.

        Returns:
            int: The number of dimensions (always 2 for a 2D vector).
        """
        return 2

    def __getitem__(self, index: int) -> float:
        """
        Get the coordinate at the given index.

        Args:
            index (int): The index of the coordinate (0 for x, 1 for y).

        Returns:
            float: The coordinate value.

        Raises:
            IndexError: If the index is out of range.
        """
        if index == 0:
            return self.x

        if index == 1:
            return self.y

        raise IndexError("Index out of range")
    
    def __eq__(self, other: object) -> bool:
        """
        Compare two vectors for equality.

        Args:
            other (object): The vector to compare with.

        Returns:
            bool: True if the vectors are equal, False otherwise.
        """
        if isinstance(other, Vector2D):
            return self.x == other.x and self.y == other.y
        return False

    def __add__(self, other: "Vector2D") -> "Vector2D":
        """
        Add two vectors.

        Args:
            other (Vector2D): The vector to add.

        Returns:
            Vector2D: The result of vector addition.

        Raises:
            TypeError: If the operand is not a Vector2D instance.
        """
        if not isinstance(other, Vector2D):
            raise TypeError(
                f"Unsupported operand type(s) for +: 'Vector2D' and '{type(other).__name__}'"
            )

        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vector2D") -> "Vector2D":
        """
        Subtract one vector from another.

        Args:
            other (Vector2D): The vector to subtract.

        Returns:
            Vector2D: The result of vector subtraction.

        Raises:
            TypeError: If the operand is not a Vector2D instance.
        """
        if not isinstance(other, Vector2D):
            raise TypeError(
                f"Unsupported operand type(s) for -: 'Vector2D' and '{type(other).__name__}'"
            )

        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: Union[int, float]) -> "Vector2D":
        """
        Multiply the vector by a scalar.

        Args:
            scalar (Union[int, float]): The scalar to multiply by.

        Returns:
            Vector2D: The result of scalar multiplication.

        Raises:
            TypeError: If the operand is not an int or float.
        """
        if not isinstance(scalar, (int, float)):
            raise TypeError(
                f"Unsupported operand type(s) for *: 'Vector2D' and '{type(scalar).__name__}'"
            )

        return Vector2D(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: Union[int, float]) -> "Vector2D":
        """
        Right multiply the vector by a scalar.

        Args:
            scalar (Union[int, float]): The scalar to multiply by.

        Returns:
            Vector2D: The result of scalar multiplication.
        """
        return self.__mul__(scalar)

    def __truediv__(self, scalar: Union[int, float]) -> "Vector2D":
        """
        Divide the vector by a scalar.

        Args:
            scalar (Union[int, float]): The scalar to divide by.

        Returns:
            Vector2D: The result of scalar division.

        Raises:
            TypeError: If the operand is not an int or float.
            ValueError: If division by zero is attempted.
        """
        if not isinstance(scalar, (int, float)):
            raise TypeError(
                f"Unsupported operand type(s) for /: 'Vector2D' and '{type(scalar).__name__}'"
            )

        if scalar == 0:
            raise ValueError("Division by zero is not allowed")

        return Vector2D(self.x / scalar, self.y / scalar)
