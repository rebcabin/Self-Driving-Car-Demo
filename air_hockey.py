import pygame
from pygame.color import THECOLORS

import pymunk
from pymunk.vec2d import Vec2d
from pymunk.pygame_util import draw

import time
import numpy as np
import numpy.random as rndm
from typing import List, Tuple, Callable


WIDTH = 1000
HEIGHT = 700
DEMO_STEPS = 100
DEMO_DT = 1.00
DEMO_RADIUS = 100
PADDING = 10

TOP_LEFT = (PADDING, PADDING)
BOTTOM_LEFT = (PADDING, HEIGHT - PADDING - 1)
BOTTOM_RIGHT = (WIDTH - PADDING - 1, HEIGHT - PADDING - 1)
TOP_RIGHT = Vec2d(WIDTH - PADDING - 1, PADDING)


def clear_screen(color=THECOLORS['black']):
    g_screen.fill(color)
    pygame.display.flip()


class Wall(object):

    def __init__(self,
                 left: Vec2d,
                 right: Vec2d,
                 color=THECOLORS['yellow']):
        self.left = left
        self.right = right
        self.color = color

    def draw(self):
        pass


class Puck(object):

    def __init__(self,
                 center,
                 velocity,
                 radius=DEMO_RADIUS,
                 color=THECOLORS['red'],
                 dont_fill_bit=0):

        self._original_center = center
        self._original_velocity = velocity

        self.center = center
        self.velocity = velocity
        self.radius = radius
        self.color = color
        self.dont_fill_bit = dont_fill_bit

    def reset(self):
        self.center = self._original_center
        self.velocity = self._original_velocity

    def step(self, dt: float):
        self.center += dt * self.velocity

    def draw(self):
        pygame.draw.circle(g_screen,
                           self.color,
                           self.center.int_tuple,
                           self.radius,
                           self.dont_fill_bit)

    def animate(self, steps, dt: float):
        for i in range(steps):
            self.step(dt)
            clear_screen()
            self.draw()
            pygame.display.flip()

    def predict_a_wall_collision(self, wall: Wall, dt=1):
        p = self.center
        q, t = colinear_point_and_parameter(wall.left, wall.right, p)
        puck_drop_normal_direction = (q - p).normalized()
        puck_drop_point_on_circle = p + self.radius * puck_drop_normal_direction
        q_prime, t_prime = colinear_point_and_parameter(
            wall.left, wall.right, puck_drop_point_on_circle)
        # q_prime should be almost the same as q
        # TODO: np.testing.assert_allclose(), meanwhile, inspect in debugger.
        projected_speed = self.velocity.dot(puck_drop_normal_direction)
        distance_to_wall = (q_prime - p).length
        predicted_time = dt * distance_to_wall / projected_speed
        return predicted_time, q_prime, t_prime, wall

    def find_nearest_wall_collision(self, walls):
        predictions = [self.predict_a_wall_collision(wall) for wall in walls]
        prediction = arg_min(
            predictions,
            lambda p: p[0] if p[0] >= 0 else np.inf)
        return prediction


# Function to return a parametric function that produces all the points along
# a line:


def parametric_line(p0: Vec2d, p1: Vec2d) -> Callable:
    d = (p1 - p0).get_length()
    return lambda t: p0 + t * (p1 - p0) / d


def random_points(n):
    return [Vec2d(p[0] * WIDTH, p[1] * HEIGHT) for p in rndm.random((n, 2))]


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
            Vec2d(pad, HEIGHT - pad - 1),
            Vec2d(WIDTH - pad - 1, HEIGHT - pad - 1),
            Vec2d(WIDTH - pad - 1, pad)]


def draw_points(int_tuples: List[Tuple[int, int]],
                color=THECOLORS['yellow']):
    pygame.draw.polygon(g_screen, color, int_tuples, 1)
    pygame.display.flip()


def colinear_point_and_parameter(u: Vec2d, v: Vec2d, p: Vec2d) -> \
        Tuple[Vec2d, float]:
    vmu_squared = (v - u).dot(v - u)
    t = (p - u).dot(v - u) / vmu_squared
    q = u + t * (v - u)
    return (q, t)


def draw_colinear_point_and_param(u=Vec2d(10, HEIGHT - 10 - 1),
                                  v=Vec2d(WIDTH - 10 - 1, HEIGHT - 10 - 1),
                                  p=Vec2d(WIDTH / 2 + DEMO_STEPS - 1,
                                          HEIGHT / 2 + (DEMO_STEPS - 1) / 2),
                                  point_color=THECOLORS['white'],
                                  line_color=THECOLORS['cyan']):
    spot_radius = 9
    dont_fill_bit = 0
    q, t = colinear_point_and_parameter(u, v, p)
    pygame.draw.circle(g_screen, point_color, p.int_tuple, spot_radius,
                       dont_fill_bit)
    # pygame.draw.line(screen, point_color, q.int_tuple, q.int_tuple)
    pygame.draw.line(g_screen, line_color, p.int_tuple, q.int_tuple)
    pygame.display.flip()


def pairwise(ls, fn):
    result = []
    for i in range(len(ls) - 1):
        temp = fn(ls[i], ls[i + 1])
        result.append(temp)
    return result


def pairwise_toroidal(ls, fn):
    return pairwise(ls + [ls[0]], fn)


def draw_cage():
    draw_points([p.int_tuple for p in screen_cage()], THECOLORS['green'])


def main():
    global g_screen
    set_up_screen()
    demo_hull(0.75)
    demo_cage()


def arg_min(things, criterion):
    so_far = np.inf
    the_one = None
    for thing in things:
        c = criterion(thing)
        if c < so_far:
            the_one = thing
            so_far = c
    return the_one


def demo_cage(pause=1.5):
    clear_screen()
    puck = Puck(center=Vec2d(WIDTH / 2, HEIGHT /2), velocity=Vec2d(1, 1/2))
    puck.animate(DEMO_STEPS, DEMO_DT)
    draw_cage()
    draw_perps_to_cage()
    cage = screen_cage()
    walls = pairwise_toroidal(cage, Wall)
    predictions = [puck.predict_a_wall_collision(wall) for wall in walls]
    # find smallest non-negative predicted collision
    prediction = arg_min(
        predictions,
        lambda p: p[0] if p[0] >= 0 else np.inf)
    # puck.animate(3 * DEMO_STEPS, DEMO_DT)
    time.sleep(pause)


def draw_perps_to_cage():
    top_left = Vec2d(TOP_LEFT)
    bottom_left = Vec2d(BOTTOM_LEFT)
    bottom_right = Vec2d(BOTTOM_RIGHT)
    top_right = Vec2d(TOP_RIGHT)
    draw_colinear_point_and_param(bottom_left, bottom_right)
    draw_colinear_point_and_param(top_left, top_right)
    draw_colinear_point_and_param(top_left, bottom_left)
    draw_colinear_point_and_param(top_right, bottom_right)


def demo_hull(pause=7.0):
    clear_screen()
    puck = Puck(center=Vec2d(WIDTH / 2, HEIGHT /2), velocity=Vec2d(1, 1/2))
    puck.animate(DEMO_STEPS, DEMO_DT)
    hull = convex_hull(random_points(15))
    draw_points(hull)
    pairwise_toroidal(hull,
             lambda p0, p1: draw_colinear_point_and_param(
                 Vec2d(p0), Vec2d(p1), line_color=THECOLORS['purple']))
    time.sleep(pause)


def set_up_screen(pause=0.75):
    global g_screen
    pygame.init()
    g_screen = pygame.display.set_mode((WIDTH, HEIGHT))
    # clock = pygame.time.Clock()
    g_screen.set_alpha(None)
    time.sleep(pause)


class GameState(object):

    def __init__(self):
        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0, 0)
        self.create_car(100, 100, 0.5)
        self.num_steps = 0

        walls = [
            pymunk.Segment(self.space.static_body, TOP_LEFT, BOTTOM_LEFT, 1),
            pymunk.Segment(self.space.static_body, BOTTOM_LEFT, BOTTOM_RIGHT, 1),
            pymunk.Segment(self.space.static_body, BOTTOM_RIGHT, TOP_RIGHT, 1),
            pymunk.Segment(self.space.static_body, TOP_RIGHT, TOP_LEFT, 1),
        ]
        for s in walls:
            s.friction = 1.
            s.group = 1
            s.collision_type = 1
            s.color = THECOLORS['red']
        self.space.add(walls)

        self.obstacles = []  # Start with singleton. Add more if interesting.
        self.obstacles.append(self.create_obstacle(200, 350, 100))
        self.create_cat()

    def create_obstacle(self, x, y, r):
        c_body = pymunk.Body(pymunk.inf, pymunk.inf)
        c_shape = pymunk.Circle(c_body, r)
        c_shape.elasticity = 1.0
        c_body.position = x, y
        c_shape.color = THECOLORS["blue"]
        self.space.add(c_body, c_shape)
        return c_body

    def create_car(self, x, y, r):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.car_body = pymunk.Body(1, inertia)
        self.car_body.position = x, y
        self.car_shape = pymunk.Circle(self.car_body, 25)
        self.car_shape.color = THECOLORS["green"]
        self.car_shape.elasticity = 1.0
        self.car_body.angle = r
        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.car_body.apply_impulse(driving_direction)
        self.space.add(self.car_body, self.car_shape)

    def create_cat(self):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.cat_body = pymunk.Body(1, inertia)
        self.cat_body.position = 50, HEIGHT - 100
        self.cat_shape = pymunk.Circle(self.cat_body, 30)
        self.cat_shape.color = THECOLORS["orange"]
        self.cat_shape.elasticity = 1.0
        self.cat_shape.angle = 0.5
        direction = Vec2d(1, 0).rotated(self.cat_body.angle)
        self.space.add(self.cat_body, self.cat_shape)

    def frame_step(self):
        if self.num_steps % 100 == 0:
            self.move_obstacles()

        if self.num_steps % 5 == 0:
            self.move_cat()

        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.car_body.velocity = 100 * driving_direction

        g_screen.fill(THECOLORS["black"])
        draw(g_screen, self.space)
        self.space.step(1./10)
        pygame.display.flip()

    def move_obstacles(self):
        # Randomly move obstacles around.
        for obstacle in self.obstacles:
            speed = np.random.randint(1, 5)
            direction = Vec2d(1, 0).rotated(
                self.car_body.angle + np.random.randint(-2, 2))
            obstacle.velocity = speed * direction

    def move_cat(self):
        speed = np.random.randint(20, 200)
        self.cat_body.angle -= np.random.randint(-1, 1)
        direction = Vec2d(1, 0).rotated(self.cat_body.angle)
        self.cat_body.velocity = speed * direction


if __name__ == "__main__":
    main()
    game_state = GameState()
    for _ in range(10000):
        game_state.frame_step()
