import math

def round_up(number: float) -> int:
    '''
    This function is going to take a float and return an integer
    by rounding up to the nearest int using the `math` library

    Args:
        number (float): original floating point number to round up

    Returns:
        number (int): original number rounded up to the nearest integer
    '''
    return math.ceil(number)
