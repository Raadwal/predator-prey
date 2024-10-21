from typing import Union

def scale_value(value: Union[int, float],
                input_min: Union[int, float],
                input_max: Union[int, float],
                output_min: Union[int, float],
                output_max: Union[int, float]) -> Union[int, float]:
    """
    Scales a value from an input range to an output range.

    Args:
        value (Union[int, float]): The value to scale.
        input_min (Union[int, float]): The minimum of the input range.
        input_max (Union[int, float]): The maximum of the input range.
        output_min (Union[int, float]): The minimum of the output range.
        output_max (Union[int, float]): The maximum of the output range.

    Returns:
        Union[int, float]: The scaled value, constrained to the output range.
    """
    # Handle cases where input range is zero to avoid division by zero
    if input_min == input_max:
        raise ValueError("Input min and max cannot be the same")

    # Scaling the value
    input_range = input_max - input_min
    output_range = output_max - output_min
    scaled_value = (value - input_min) * output_range / input_range + output_min

    # Clamping the value to the output range
    return max(min(scaled_value, output_max), output_min)