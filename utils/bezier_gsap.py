import json
import os

with open(os.path.join(os.path.dirname(__file__), "lookup.json")) as file:
    lookup = json.load(file)


def calculate_y_gsap(x: float) -> float:
    index = int(x * len(lookup))
    point = lookup[index] if index < len(lookup) else lookup[-1]
    y = point["y"] + (x - point["x"]) / point["cx"] * point["cy"]

    return y


def calculate_x_gsap(y: float) -> float:
    prev_point = lookup[0]
    for point in lookup:
        if y <= point["y"]:
            break
        prev_point = point

    point = prev_point
    x = point["x"] + (y - point["y"]) / point["cy"] * point["cx"]

    return x
