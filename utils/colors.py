from typing import List

import numpy as np


def generate_diverse_colors(num_colors: int) -> List[tuple[int, int, int]]:
    grid_size = int(np.ceil(num_colors ** (1 / 3)))
    colors = set()

    while len(colors) < num_colors:
        for r in np.linspace(0, 255, grid_size):
            for g in np.linspace(0, 255, grid_size):
                for b in np.linspace(0, 255, grid_size):
                    if r + g + b > 30:  # Exclude dark colors
                        colors.add((int(r), int(g), int(b)))
                    if len(colors) >= num_colors:
                        break
                if len(colors) >= num_colors:
                    break
            if len(colors) >= num_colors:
                break
        grid_size += 1  # Increase grid density if not enough colors

    colors = list(colors)

    return colors[:num_colors]
