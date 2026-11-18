"""Rounded tire outer-envelope samples. MuJoCo uses the convex hull for collisions."""

import math


def rounded_envelope_vertices(radius, half_width, circumferential, crown):
    if not (radius > half_width > 0 and circumferential >= 12 and crown >= 8):
        raise ValueError("Invalid rounded tire dimensions or sampling")
    if circumferential % 4 or crown % 4:
        raise ValueError("Sampling counts must be divisible by four to preserve axis extrema")
    major = radius - half_width
    points = []
    for i in range(circumferential):
        a = 2 * math.pi * i / circumferential
        for j in range(crown):
            b = 2 * math.pi * j / crown
            rho = major + half_width * math.cos(b)
            points.extend((rho * math.cos(a), half_width * math.sin(b), rho * math.sin(a)))
    return points
