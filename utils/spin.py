def range(length: int) -> tuple[float, float]:
    # https://github.com/Pointauc/pointauc_frontend/blob/310893f1b9e58068a9ece793a4b71a6cd11baea1/src/components/BaseWheel/BaseWheel.tsx#L147
    min_range = round(length * 270 / 360) * 360
    max_range = min_range + 360

    return float(min_range), float(max_range)


def range_with_angle(angle: float, length: int) -> float:
    min_range, _ = range(length)

    return min_range + angle
