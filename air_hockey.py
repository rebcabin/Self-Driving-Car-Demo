import pygame
from pygame.color import THECOLORS

import pymunk
from pymunk.vec2d import Vec2d
from pymunk.pygame_util import draw

import time
import numpy as np
import numpy.random as rndm
from typing import List, Tuple, Callable


SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
DEMO_STEPS = 100
DEMO_DT = 1.00
DEMO_RADIUS = 100
PADDING = 10

TOP_LEFT = (PADDING, PADDING)
BOTTOM_LEFT = (PADDING, SCREEN_HEIGHT - PADDING - 1)
BOTTOM_RIGHT = (SCREEN_WIDTH - PADDING - 1, SCREEN_HEIGHT - PADDING - 1)
TOP_RIGHT = Vec2d(SCREEN_WIDTH - PADDING - 1, PADDING)


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
                 mass,
                 radius=DEMO_RADIUS,
                 color=THECOLORS['red'],
                 dont_fill_bit=0):

        self._original_center = center
        self._original_velocity = velocity

        self.center = center
        self.velocity = velocity
        self.mass = mass
        self.radius = radius
        self.color = color
        self.dont_fill_bit = dont_fill_bit

    def momentum(self):
        return self.mass * self.velocity

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

    def step_many(self, steps, dt: float):
        for i in range(steps):
            self.step(dt)

    def predict_a_wall_collision(self, wall: Wall, dt=1):
        p = self.center
        q, t = collinear_point_and_parameter(wall.left, wall.right, p)
        puck_drop_normal_direction = (q - p).normalized()
        puck_drop_point_on_circle = p + self.radius * puck_drop_normal_direction
        q_prime, t_prime = collinear_point_and_parameter(
            wall.left, wall.right, puck_drop_point_on_circle)
        # q_prime should be almost the same as q
        # TODO: np.testing.assert_allclose(), meanwhile, inspect in debugger.
        projected_speed = self.velocity.dot(puck_drop_normal_direction)
        distance_to_wall = (q_prime - p).length
        predicted_time = dt * distance_to_wall / projected_speed \
            if projected_speed != 0 else np.inf
        return predicted_time, q_prime, t_prime, wall

    def overlapping_disaster(self, other: 'Puck', dt=1):
        _, d = self._get_relative_distance(other)
        return d < 0

    def _get_relative_distance(self, other: 'Puck'):
        p = self.center
        q = other.center
        dp = q - p  # other's position in my inertial frame
        d = dp.length - self.radius - other.radius
        return dp, d

    def predict_a_puck_collision(self, other: 'Puck', dt=1):
        """See https://goo.gl/jQik91 for forward-references as strings."""
        dp, d = self._get_relative_distance(other)
        # other's velocity in my inertial frame
        dv = other.velocity - self.velocity
        # normal component of other's velocity in my inertial frame
        n = dp.normalized()
        dv_n = dv.dot(n)
        if d < 0:
            # they're overlapped NOW! Too late!
            tau_impact = -np.inf
        else:
            if dv_n < 0:
                # other is heading toward me
                tau_impact = dt * d / dv_n
            elif dv_n == 0:
                # other is going exactly parallel to me
                tau_impact = np.inf
                # TODO: test for glancing or continuous contact
            else:
                # other is heading away from me
                tau_impact = np.inf
        return tau_impact, other, dp, n, dv, dv_n

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
    return [Vec2d(p[0] * SCREEN_WIDTH, p[1] * SCREEN_HEIGHT) for p in rndm.random((n, 2))]


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
            Vec2d(pad, SCREEN_HEIGHT - pad - 1),
            Vec2d(SCREEN_WIDTH - pad - 1, SCREEN_HEIGHT - pad - 1),
            Vec2d(SCREEN_WIDTH - pad - 1, pad)]


def draw_points(int_tuples: List[Tuple[int, int]],
                color=THECOLORS['yellow']):
    pygame.draw.polygon(g_screen, color, int_tuples, 1)
    pygame.display.flip()


def collinear_point_and_parameter(u: Vec2d, v: Vec2d, p: Vec2d) -> \
        Tuple[Vec2d, float]:
    vmu_squared = (v - u).dot(v - u)
    t = (p - u).dot(v - u) / vmu_squared
    q = u + t * (v - u)
    return (q, t)


def draw_collinear_point_and_param(
        u=Vec2d(10, SCREEN_HEIGHT - 10 - 1),
        v=Vec2d(SCREEN_WIDTH - 10 - 1, SCREEN_HEIGHT - 10 - 1),
        p=Vec2d(SCREEN_WIDTH / 2 + DEMO_STEPS - 1,
                SCREEN_HEIGHT / 2 + (DEMO_STEPS - 1) / 2),
        point_color=THECOLORS['white'],
        line_color=THECOLORS['cyan']):
    spot_radius = 9
    dont_fill_bit = 0
    q, t = collinear_point_and_parameter(u, v, p)
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


def arg_min(things, criterion):
    so_far = np.inf
    the_one = None
    for thing in things:
        c = criterion(thing)
        if c < so_far:
            the_one = thing
            so_far = c
    return the_one


def demo_cage(pause=0.75):
    clear_screen()

    me = Puck(center=Vec2d(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2),
                velocity=random_velocity(),
                mass=100,
                radius=42)

    them = Puck(center=Vec2d(SCREEN_WIDTH / 1.5, SCREEN_HEIGHT / 2.5),
                velocity=random_velocity(),
                mass=100,
                radius=79,
                color=THECOLORS['green'])

    clear_screen()
    me.draw()
    them.draw()
    me.step_many(DEMO_STEPS, DEMO_DT)
    them.step_many(DEMO_STEPS, DEMO_DT)
    me.draw()
    them.draw()
    pygame.display.flip()

    draw_cage()

    draw_perps_to_cage(me)
    draw_perps_to_cage(them)

    cage = screen_cage()

    walls = pairwise_toroidal(cage, Wall)
    wall_collision_predictions = \
        [me.predict_a_wall_collision(wall) for wall in walls]
    # find smallest non-negative predicted collision
    wall_prediction = arg_min(
        wall_collision_predictions,
        lambda p: p[0] if p[0] >= 0 else np.inf)
    # puck.animate(3 * DEMO_STEPS, DEMO_DT)
    time.sleep(pause)


def draw_perps_to_cage(puck: Puck):
    top_left = Vec2d(TOP_LEFT)
    bottom_left = Vec2d(BOTTOM_LEFT)
    bottom_right = Vec2d(BOTTOM_RIGHT)
    top_right = Vec2d(TOP_RIGHT)
    p = puck.center
    draw_collinear_point_and_param(bottom_left, bottom_right, p)
    draw_collinear_point_and_param(top_left, top_right, p)
    draw_collinear_point_and_param(top_left, bottom_left, p)
    draw_collinear_point_and_param(top_right, bottom_right, p)


def demo_hull(pause=7.0):
    clear_screen()
    puck = Puck(
        center=Vec2d(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2),
        velocity=Vec2d(1, 1/2),
        mass = 100)
    puck.step_many(DEMO_STEPS, DEMO_DT)
    hull = convex_hull(random_points(15))
    draw_points(hull)
    pairwise_toroidal(hull,
             lambda p0, p1: draw_collinear_point_and_param(
                 Vec2d(p0), Vec2d(p1), line_color=THECOLORS['purple']))
    time.sleep(pause)


def set_up_screen(pause=0.75):
    global g_screen
    pygame.init()
    g_screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    # clock = pygame.time.Clock()
    g_screen.set_alpha(None)
    time.sleep(pause)


def random_velocity():
    speed = np.random.randint(1, 5)
    direction = Vec2d(1, 0).rotated(np.random.randint(-2, 2))
    result = speed * direction
    return result


def create_ball(x, y, r, m, color, e=1.0):
    body = pymunk.Body(
        mass=m,
        moment=pymunk.moment_for_circle(
            mass=m,
            inner_radius=0,
            outer_radius=r,
            offset=(0, 0)))
    body.position = x, y
    body.velocity = random_velocity()
    shape = pymunk.Circle(body=body, radius=r)
    shape.color = color
    shape.elasticity = e
    return body, shape


class GameState(object):

    def __init__(self):
        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0, 0)

        self.create_walls()
        self.create_obstacle(x=200, y=350, r=DEMO_RADIUS)
        self.create_cat(x=700, y=SCREEN_HEIGHT - PADDING - 100, r=30)
        self.create_car(x=100, y=100, r=25)

    def create_walls(self):
        walls = [
            pymunk.Segment(self.space.static_body, TOP_LEFT, BOTTOM_LEFT, 1),
            pymunk.Segment(self.space.static_body, BOTTOM_LEFT, BOTTOM_RIGHT,
                           1),
            pymunk.Segment(self.space.static_body, BOTTOM_RIGHT, TOP_RIGHT, 1),
            pymunk.Segment(self.space.static_body, TOP_RIGHT, TOP_LEFT, 1),
        ]
        for s in walls:
            s.friction = 1.
            s.elasticity = 0.95
            s.group = 1
            s.collision_type = 1
            s.color = THECOLORS['red']
        self.space.add(walls)

    def add_ball(self, x, y, r, m, c):
        body, shape = create_ball(x, y, r, m, c)
        self.space.add(body, shape)

    def create_obstacle(self, x, y, r, m=100):
        self.add_ball(x, y, r, m, THECOLORS['blue'])

    def create_car(self, x, y, r, m=1):
        self.add_ball(x, y, r, m, THECOLORS['green'])

    def create_cat(self, x, y, r, m=1):
        self.add_ball(x, y, r, m, THECOLORS['orange'])

    def frame_step(self):
        # TODO: no easy way to reset the angle marker after a collision.
        g_screen.fill(THECOLORS["black"])
        draw(g_screen, self.space)
        self.space.step(1./10)
        pygame.display.flip()


def demo_classic(steps=500):
    game_state = GameState()
    for _ in range(steps):
        game_state.frame_step()


def main():
    global g_screen
    set_up_screen()
    # demo_hull(0.75)
    demo_cage(pause=3.0)
    # demo_classic(steps=3000)


if __name__ == "__main__":
    main()
