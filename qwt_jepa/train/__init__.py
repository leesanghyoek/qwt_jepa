import math


def ema_momentum(step: int, total_steps: int, start: float = 0.996, end: float = 1.0) -> float:
    """Lich momentum EMA: tang tu `start` -> `end` theo cosine trong `total_steps` buoc."""
    step = max(0, min(step, total_steps))
    if total_steps <= 0:
        return end
    return end - (end - start) * 0.5 * (1.0 + math.cos(math.pi * step / total_steps))
