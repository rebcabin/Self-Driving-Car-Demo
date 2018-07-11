import pygame
from pygame.color import THECOLORS
from pymunk.vec2d import Vec2d

width = 1000
height = 700
pygame.init()
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()
screen.set_alpha(None)


def clear_screen(color=THECOLORS['black']):
    screen.fill(color)
    pygame.display.flip()


def draw_puck(p_0=Vec2d(width / 2, height / 2), puck_color=THECOLORS['red']):
    dont_fill_bit = 0
    clear_screen()
    r = 100
    pygame.draw.circle(screen, puck_color, p_0.int_tuple, r, dont_fill_bit)
    pygame.display.flip()


def animate_puck(
        steps=100,
        dt=1,
        p_0=Vec2d(width / 2, height / 2),
        v_0=Vec2d(1, 1 / 2),
        puck_radius=100,
        puck_color=THECOLORS['red'],
        dont_fill_bit=0):
    for i in range(steps):
        clear_screen()
        p_i = p_0 + (i * dt * v_0)
        pygame.draw.circle(screen,
                           puck_color,
                           p_i.int_tuple,
                           puck_radius,
                           dont_fill_bit)
        pygame.display.flip()


## PREDICT WALL STRIKES

# Function to return a parametric function that produces all the points along
# a line:


from typing import List, Tuple, Callable


def parametric_line(p0: Vec2d, p1: Vec2d) -> Callable:
    d = (p1 - p0).get_length()
    return lambda t: p0 + t * (p1 - p0) / d


import numpy.random as rndm


def random_points(n):
    return [Vec2d(p[0] * width, p[1] * height) for p in rndm.random((n, 2))]


# https://goo.gl/PR24H2
# https://goo.gl/x3dEM6

def convex_hull(points: List[Vec2d]) -> List[Tuple[int, int]]:
    """Computes the convex hull of a set of 2D points.

    Input: an iterable sequence of Vec2ds(x, y) representing the points.
    Output: a list of integer vertices of the convex hull in counter-clockwise
    order, starting from the vertex with the lexicographically smallest
    coordinates. Implements Andrew's monotone chain algorithm.
    O(n log n) complexity in time."""

    # Sort the points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    points = sorted(set([p.int_tuple for p in points]))

    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(points) <= 1:
        return points

    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the
    # beginning of the other list.
    return lower[:-1] + upper[:-1]


def screen_cage(pad=10):
    return [Vec2d(pad, pad),
            Vec2d(pad, height - pad - 1),
            Vec2d(width - pad - 1, height - pad - 1),
            Vec2d(width - pad - 1, pad)]


def draw_points(int_tuples: List[Tuple[int, int]],
                color=THECOLORS['yellow']):
    pygame.draw.polygon(screen, color, int_tuples, 1)
    pygame.display.flip()


animate_puck()
# draw_points(convex_hull(random_points(15)))
draw_points([p.int_tuple for p in screen_cage()], THECOLORS['green'])


def colinear_point_and_parameter(u: Vec2d, v: Vec2d, p: Vec2d) -> \
        Tuple[Vec2d, float]:
    vmu_squared = (v - u).dot(v - u)
    t = (p - u).dot(v - u) / vmu_squared
    q = u + t * (v - u)
    return (q, t)


def draw_col_pt_param(u=Vec2d(10, height - 10 - 1),
                      v=Vec2d(width - 10 - 1, height - 10 - 1),
                      p=Vec2d(width / 2 + 99, height / 2 + 99 / 2),
                      point_color=THECOLORS['white'],
                      line_color=THECOLORS['cyan']):
    spot_radius = 9
    dont_fill_bit = 0
    q, t = colinear_point_and_parameter(u, v, p)
    pygame.draw.circle(screen, point_color, p.int_tuple, spot_radius,
                       dont_fill_bit)
    # pygame.draw.line(screen, point_color, q.int_tuple, q.int_tuple)
    pygame.draw.line(screen, line_color, p.int_tuple, q.int_tuple)
    pygame.display.flip()


pad = 10


top_left     = Vec2d(pad,             pad)
bottom_left  = Vec2d(pad,             height - pad - 1)
bottom_right = Vec2d(width - pad - 1, height - pad - 1)
top_right    = Vec2d(width - pad - 1, pad)


draw_col_pt_param(bottom_left, bottom_right)
draw_col_pt_param(top_left, top_right)
draw_col_pt_param(top_left, bottom_left)
draw_col_pt_param(top_right, bottom_right)


import numpy as np


def predict_hit(line_p0: Vec2d,
                line_p1: Vec2d,
                puck_center: Vec2d = Vec2d(width / 2, height / 2),
                puck_radius: float = 100.0,
                puck_vel: Vec2d = Vec2d(1, 1 / 2),
                dt: float = 1.0):
    q, t = colinear_point_and_parameter(line_p0, line_p1, puck_center)
    puck_drop_vector = q - p
    puck_drop_length_squared = puck_drop_vector.dot(puck_drop_vector)
    puck_drop_length = np.sqrt(puck_drop_length_squared)
    puck_drop_normal = puck_drop_vector / puck_drop_length
    puck_drop_on_circle = puck_radius * puck_drop_normal
    q_prime, t_prime = colinear_point_and_parameter(line_p0, line_p1,
                                                    puck_drop_on_circle)
    pass


import time


time.sleep(3.0)


