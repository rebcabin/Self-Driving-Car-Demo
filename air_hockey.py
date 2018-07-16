import sys

import pygame
from pygame.color import THECOLORS

import pymunk
from pymunk.vec2d import Vec2d
from pymunk.pygame_util import draw

import time
import numpy as np
import numpy.random as rndm
from typing import List, Tuple, Callable, Dict, Any

import pprint

pp = pprint.PrettyPrinter(indent=2)

#   ___             _            _
#  / __|___ _ _  __| |_ __ _ _ _| |_ ___
# | (__/ _ \ ' \(_-<  _/ _` | ' \  _(_-<
#  \___\___/_||_/__/\__\__,_|_||_\__/__/


SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
DEMO_STEPS = 100
DEMO_DT = 1.00
DEMO_RADIUS = 100
PADDING = 10

SPOT_RADIUS = 9

TOP_LEFT = (PADDING, PADDING)
BOTTOM_LEFT = (PADDING, SCREEN_HEIGHT - PADDING - 1)
BOTTOM_RIGHT = (SCREEN_WIDTH - PADDING - 1, SCREEN_HEIGHT - PADDING - 1)
TOP_RIGHT = Vec2d(SCREEN_WIDTH - PADDING - 1, PADDING)


# TODO: IDs might be best as uuids.
# TODO: Message equality can be optimized.


# __   ___     _             _ _____ _
# \ \ / (_)_ _| |_ _  _ __ _| |_   _(_)_ __  ___
#  \ V /| | '_|  _| || / _` | | | | | | '  \/ -_)
#   \_/ |_|_|  \__|\_,_\__,_|_| |_| |_|_|_|_\___|


class VirtualTime(int):
    """Establishes a type for Virtual Time; it's an int."""


#  ___                         ___ ___
# | _ \_ _ ___  __ ___ ______ |_ _|   \
# |  _/ '_/ _ \/ _/ -_|_-<_-<  | || |) |
# |_| |_| \___/\__\___/__/__/ |___|___/


class ProcessID(str):
    """Establishes a type for Process ID; it's a string."""


#  __  __                            ___          _
# |  \/  |___ ______ __ _ __ _ ___  | _ ) ___  __| |_  _
# | |\/| / -_|_-<_-</ _` / _` / -_) | _ \/ _ \/ _` | || |
# |_|  |_\___/__/__/\__,_\__, \___| |___/\___/\__,_|\_, |
#                        |___/                      |__/


class Body(Dict):
    """Establishes a type for message bodies; they're Dicts."""


#  _____ _              _                           _
# |_   _(_)_ __  ___ __| |_ __ _ _ __  _ __  ___ __| |
#   | | | | '  \/ -_|_-<  _/ _` | '  \| '_ \/ -_) _` |
#   |_| |_|_|_|_\___/__/\__\__,_|_|_|_| .__/\___\__,_|
#                                     |_|


# See https://goo.gl/hiwMgJ


class ComparableMixin(object):
    def _compare(self, other, method):
        try:
            return method(self._cmpkey(), other._cmpkey())
        except (AttributeError, TypeError):
            # _cmpkey not implemented, or return different type,
            # so I can't compare with "other".
            return NotImplemented

    def __lt__(self, other):
        return self._compare(other, lambda s, o: s < o)

    def __le__(self, other):
        return self._compare(other, lambda s, o: s <= o)

    def __eq__(self, other):
        return self._compare(other, lambda s, o: s == o)

    def __ge__(self, other):
        return self._compare(other, lambda s, o: s >= o)

    def __gt__(self, other):
        return self._compare(other, lambda s, o: s > o)

    def __ne__(self, other):
        return self._compare(other, lambda s, o: s != o)


class Timestamped(ComparableMixin):
    """Establishes a type for timestamped objects:
    messages, states, logical processes."""

    def __init__(self, vt: VirtualTime):
        self.vt = vt

    def _cmpkey(self):
        return self.vt


#  ___             _     __  __
# | __|_ _____ _ _| |_  |  \/  |___ ______ __ _ __ _ ___
# | _|\ V / -_) ' \  _| | |\/| / -_|_-<_-</ _` / _` / -_)
# |___|\_/\___|_||_\__| |_|  |_\___/__/__/\__,_\__, \___|
#                                              |___/


class EventMessage(Timestamped):
    """The vt field is an innovation over classical time warp. It is merely a
    convenient to make the common case easy. Positive messages have
    vt=receivetime by default, negative have vt=sendtime. That makes easy the
    common case of inserting positive messages into input queues and negative
    messages into output queues. When a negative message is inserted into an
    input queue, its vt must be switched to the receive time. Likewise, when a
    positive message is inserted into an output queue, its vt must be switched
    to send time."""

    def __init__(self,
                 sender: ProcessID, sendtime: VirtualTime,
                 receiver: ProcessID, receivetime: VirtualTime,
                 sign: bool, body: Body):
        if receivetime <= sendtime:
            raise ValueError(f"receive time {receivetime} must be strictly "
                             f"greater than send time {sendtime}")
        super().__init__(vt=receivetime if sign else sendtime)
        self.sender = sender
        self.send_time = sendtime
        self.receiver = receiver
        self.receive_time = receivetime
        self.sign = sign
        self.body = body

    def __eq__(self, other: 'EventMessage'):
        """Check equality for all attributes EXCEPT algebraic sign.
        TODO: optimize with hashes or uuids."""
        return self.sender == other.sender and \
               self.send_time == other.send_time and \
               self.receiver == other.receiver and \
               self.receive_time == other.receive_time and \
               self.body == other.body

    def __ne__(self, other: 'EventMessage'):
        """Don't use default timestamp comparable for !=."""
        return not self == other


#  _______      _____ _        _
# |_   _\ \    / / __| |_ __ _| |_ ___
#   | |  \ \/\/ /\__ \  _/ _` |  _/ -_)
#   |_|   \_/\_/ |___/\__\__,_|\__\___|


class State(EventMessage):
    def __init__(self,
                 sender: ProcessID, sendtime: VirtualTime,
                 body: Body):
        """Modeled as a negative event message from self to self with indeterminate
        (infinite) receive time. It's negative, so its timestamp is the
        sendtime, just like an output negative message."""
        super().__init__(sender=sender,
                         sendtime=sendtime,
                         receiver=sender,
                         receivetime=sys.maxsize,
                         sign=False,
                         body=body)


#   ___                      __  __
#  / _ \ _  _ ___ _ _ _  _  |  \/  |___ ______ __ _ __ _ ___
# | (_) | || / -_) '_| || | | |\/| / -_|_-<_-</ _` / _` / -_)
#  \__\_\\_,_\___|_|  \_, | |_|  |_\___/__/__/\__,_\__, \___|
#                     |__/                         |___/


#  _______      _____
# |_   _\ \    / / _ \ _  _ ___ _  _ ___
#   | |  \ \/\/ / (_) | || / -_) || / -_)
#   |_|   \_/\_/ \__\_\\_,_\___|\_,_\___|


class TWQueue(object):
    """Implements timestamp-ordered, vt-cursored queue with annihilation.
    TODO: optimize with priority queue or red-black tree."""

    def __init__(self):
        self.items = []
        self.vt = -sys.maxsize
        self.rollback = False
        self.annihilation = False

    def vts(self):
        """For debugging"""
        return [i.vt for i in self.items]

    def insert(self, item: Timestamped):
        self.annihilation = False
        # ROLLBACK!
        if item.vt <= self.vt:
            self.rollback = True
            self.vt = item.vt
        i = -sys.maxsize
        top = len(self.items)
        for i in range(top):
            if item.vt > self.items[i].vt:
                break
            if item.vt == self.items[i].vt:
                for j in range(i, top):
                    if (item == self.items[j] and
                            hasattr(item, 'sign') and
                            item.sign == (not self.items[j].sign)):
                        self.annihilation = True
                        del self.items[j]
                        return
                i = j
        self.items.insert(i, item)


#  ___ _        _          ___
# / __| |_ __ _| |_ ___   / _ \ _  _ ___ _  _ ___
# \__ \  _/ _` |  _/ -_) | (_) | || / -_) || / -_)
# |___/\__\__,_|\__\___|  \__\_\\_,_\___|\_,_\___|


class StateQueue(TWQueue):
    def __init__(self):
        super().__init__()


#   ___       _             _      ___
#  / _ \ _  _| |_ _ __ _  _| |_   / _ \ _  _ ___ _  _ ___
# | (_) | || |  _| '_ \ || |  _| | (_) | || / -_) || / -_)
#  \___/ \_,_|\__| .__/\_,_|\__|  \__\_\\_,_\___|\_,_\___|
#                |_|


class OutputQueue(TWQueue):
    def __init__(self):
        super().__init__()

    def insert(self, message: EventMessage):
        if message.sign:
            message.vt = message.send_time
        super().insert(message)


#  ___                _      ___
# |_ _|_ _  _ __ _  _| |_   / _ \ _  _ ___ _  _ ___
#  | || ' \| '_ \ || |  _| | (_) | || / -_) || / -_)
# |___|_||_| .__/\_,_|\__|  \__\_\\_,_\___|\_,_\___|
#          |_|


class InputQueue(TWQueue):
    def __init__(self):
        super().__init__()

    def insert(self, message: EventMessage):
        if not message.sign:
            message.vt = message.receive_time
        super().insert(message)


#  _              _         _   ___
# | |   ___  __ _(_)__ __ _| | | _ \_ _ ___  __ ___ ______
# | |__/ _ \/ _` | / _/ _` | | |  _/ '_/ _ \/ _/ -_|_-<_-<
# |____\___/\__, |_\__\__,_|_| |_| |_| \___/\__\___/__/__/
#           |___/


class LogicalProcess(Timestamped):
    def __init__(self, pid: ProcessID):
        self.now = sys.maxsize
        super().__init__(self.now)
        self.iq = InputQueue()
        self.oq = OutputQueue()
        self.sq = StateQueue()
        self.pid = pid


#  ___     _           _      _        ___
# / __| __| |_  ___ __| |_  _| |___   / _ \ _  _ ___ _  _ ___
# \__ \/ _| ' \/ -_) _` | || | / -_) | (_) | || / -_) || / -_)
# |___/\__|_||_\___\__,_|\_,_|_\___|  \__\_\\_,_\___|\_,_\___|


class ScheduleQueue(TWQueue):
    pass


#  ___ _           _         _   ___
# | _ \ |_ _  _ __(_)__ __ _| | | _ \_ _ ___  __ ___ ______ ___ _ _
# |  _/ ' \ || (_-< / _/ _` | | |  _/ '_/ _ \/ _/ -_|_-<_-</ _ \ '_|
# |_| |_||_\_, /__/_\__\__,_|_| |_| |_| \___/\__\___/__/__/\___/_|
#          |__/


#  ___                                   ___
# | _ \_ _ ___  __ ___ ______ ___ _ _   / _ \ _  _ ___ _  _ ___
# |  _/ '_/ _ \/ _/ -_|_-<_-</ _ \ '_| | (_) | || / -_) || / -_)
# |_| |_| \___/\__\___/__/__/\___/_|    \__\_\\_,_\___|\_,_\___|


class LogicalProcess(object):

    def __init__(self):
        pass

    pass


#  _____     _    _       ___          _
# |_   _|_ _| |__| |___  | _ \___ __ _(_)___ _ _
#   | |/ _` | '_ \ / -_) |   / -_) _` | / _ \ ' \
#   |_|\__,_|_.__/_\___| |_|_\___\__, |_\___/_||_|
#                                |___/


# __      __    _ _
# \ \    / /_ _| | |
#  \ \/\/ / _` | | |
#   \_/\_/\__,_|_|_|


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


#  ___         _
# | _ \_  _ __| |__
# |  _/ || / _| / /
# |_|  \_,_\__|_\_\


class Puck(object):

    def __init__(self, center, velocity, mass, radius, color, dont_fill_bit=0):

        self._original_center = center
        self._original_velocity = velocity

        self.center = center
        self.velocity = velocity
        self.mass = mass
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

    def step_many(self, steps, dt: float):
        self.center += steps * dt * self.velocity

    def predict_a_wall_collision(self, wall: Wall, dt):
        p = self.center
        q, t = collinear_point_and_parameter(wall.left, wall.right, p)
        drop_normal_direction = (q - p).normalized()
        point_on_circle = p + self.radius * drop_normal_direction
        q_prime, t_prime = collinear_point_and_parameter(
            wall.left, wall.right, point_on_circle)
        # q_prime should be almost the same as q
        # TODO: np.testing.assert_allclose(...), meanwhile, inspect in debugger.
        projected_speed = self.velocity.dot(drop_normal_direction)
        distance_to_wall = (q_prime - point_on_circle).length
        # predicted step time can be negative! it is permitted!
        predicted_step_time = distance_to_wall / projected_speed / dt \
            if projected_speed != 0 else np.inf
        return {'tau': predicted_step_time,
                'puck_strike_point': point_on_circle,
                'wall_strike_point': q_prime,
                'wall_strike_parameter': t_prime,
                'wall_victim': wall}

    def predict_a_puck_collision(self, them: 'Puck', dt):
        """See https://goo.gl/jQik91 for forward-references as strings."""
        dp = them.center - self.center
        # Relative distance as a function of time, find its zero:
        #
        # Collect[{x-vx t, y-vy t}^2 - d1^2, t]
        #
        # (x^2+y^2)-d1^2 + t (-2 x vx-2 y vy) + t^2 (vx^2 + vy^2)
        # \_____ ______/     \______ _______/       \_____ _____/
        #       v                   v                     v
        #       c                   b                     a
        #
        dv = self.velocity - them.velocity
        a = dv.get_length_sqrd()
        b = -2 * dp.dot(dv)
        d1 = self.radius + them.radius
        c = dp.get_length_sqrd() - (d1 * d1)
        disc = (b * b) - (4 * a * c)
        gonna_hit = False
        tau_impact_steps = np.inf
        if disc >= 0 and a != 0:
            sdisc = np.sqrt(disc)
            tau1 = (-b + sdisc) / (2 * a)
            tau2 = (-b - sdisc) / (2 * a)
            if tau1 >= 0 and tau2 >= 0:
                tau_impact_steps = min(tau1, tau2) / dt
                gonna_hit = True
            else:
                tau_impact_steps = max(tau1, tau2) / dt
                gonna_hit = tau1 >= 0 or tau2 >= 0

        return {'tau': tau_impact_steps,
                'puck_victim': them,
                'gonna_hit': gonna_hit}


#   ___                   _              ___     _       _ _   _
#  / __|___ ___ _ __  ___| |_ _ _ _  _  | _ \_ _(_)_ __ (_) |_(_)_ _____ ___
# | (_ / -_) _ \ '  \/ -_)  _| '_| || | |  _/ '_| | '  \| |  _| \ V / -_|_-<
#  \___\___\___/_|_|_\___|\__|_|  \_, | |_| |_| |_|_|_|_|_|\__|_|\_/\___/__/
#                                 |__/


def parametric_line(p0: Vec2d, p1: Vec2d) -> Callable:
    """Returns a parametric function that produces all the points along an oriented
    line segment from point p0 to point p1 as the parameter varies from 0 to 1.
    Parameter values outside [0, 1] produce points along the infinite
    continuation of the line segment."""
    d = (p1 - p0).get_length()
    return lambda t: p0 + t * (p1 - p0) / d


def random_points(n):
    """Produces randomly chosen points inside the screen (TODO: possibly
    off-by-one on the right and the bottom)"""
    return [Vec2d(p[0] * SCREEN_WIDTH, p[1] * SCREEN_HEIGHT)
            for p in rndm.random((n, 2))]


def convex_hull(points: List[Vec2d]) -> List[Tuple[int, int]]:
    """Computes the convex hull of a set of 2D points.

    Input: an iterable sequence of Vec2d(x, y) representing the points. Output:
    a list of integer vertices of the convex hull in counter-clockwise order,
    starting from the vertex with the lexicographically smallest coordinates.
    Implements Andrew's monotone chain algorithm. O(n log n) complexity in
    time.

    See https://goo.gl/PR24H2 and https://goo.gl/x3dEM6"""

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


def collinear_point_and_parameter(u: Vec2d, v: Vec2d, p: Vec2d) -> \
        Tuple[Vec2d, float]:
    vmu_squared = (v - u).dot(v - u)
    t = (p - u).dot(v - u) / vmu_squared
    q = u + t * (v - u)
    return (q, t)


def rotate_seq(pt, c, s):
    x = pt[0]
    y = pt[1]
    return (c * x - s * y,
            s * x + c * y)


def translate_seq(pt, x_prime, y_prime):
    x = pt[0]
    y = pt[1]
    return (x + x_prime, y + y_prime)


def scale_seq(pt, xs, ys):
    x = pt[0]
    y = pt[1]
    return (x * xs, y * ys)


def centroid_seq(pts):
    l = len(pts)
    return (sum(pt[0] for pt in pts) / l,
            sum(pt[1] for pt in pts) / l)


#  ___             _   _               _   ___     _       _ _   _
# | __|  _ _ _  __| |_(_)___ _ _  __ _| | | _ \_ _(_)_ __ (_) |_(_)_ _____ ___
# | _| || | ' \/ _|  _| / _ \ ' \/ _` | | |  _/ '_| | '  \| |  _| \ V / -_|_-<
# |_| \_,_|_||_\__|\__|_\___/_||_\__,_|_| |_| |_| |_|_|_|_|_|\__|_|\_/\___/__/


def pairwise(ls, fn):
    result = []
    for i in range(len(ls) - 1):
        temp = fn(ls[i], ls[i + 1])
        result.append(temp)
    return result


def pairwise_toroidal(ls, fn):
    return pairwise(ls + [ls[0]], fn)


def arg_min(things, criterion):
    so_far = np.inf
    the_one = None
    for thing in things:
        c = criterion(thing)
        if c < so_far:
            the_one = thing
            so_far = c
    return the_one


#  ___             _         _
# | _ \___ _ _  __| |___ _ _(_)_ _  __ _
# |   / -_) ' \/ _` / -_) '_| | ' \/ _` |
# |_|_\___|_||_\__,_\___|_| |_|_||_\__, |
#                                  |___/


def draw_int_tuples(int_tuples: List[Tuple[int, int]],
                    color=THECOLORS['yellow']):
    pygame.draw.polygon(g_screen, color, int_tuples, 1)


def draw_collinear_point_and_param(
        u=Vec2d(10, SCREEN_HEIGHT - 10 - 1),
        v=Vec2d(SCREEN_WIDTH - 10 - 1, SCREEN_HEIGHT - 10 - 1),
        p=Vec2d(SCREEN_WIDTH / 2 + DEMO_STEPS - 1,
                SCREEN_HEIGHT / 2 + (DEMO_STEPS - 1) / 2),
        point_color=THECOLORS['white'],
        line_color=THECOLORS['cyan']):
    dont_fill_bit = 0
    q, t = collinear_point_and_parameter(u, v, p)
    pygame.draw.circle(g_screen, point_color, p.int_tuple, SPOT_RADIUS,
                       dont_fill_bit)
    # pygame.draw.line(screen, point_color, q.int_tuple, q.int_tuple)
    pygame.draw.line(g_screen, line_color, p.int_tuple, q.int_tuple)


def draw_vector(p0: Vec2d, p1: Vec2d, color):
    pygame.draw.line(g_screen, color, p0.int_tuple, p1.int_tuple)
    pygame.draw.circle(g_screen, color, p1.int_tuple, SPOT_RADIUS, 0)


def draw_centered_arrow(loc, vel):
    arrow_surface = g_screen.copy()
    arrow_surface.set_alpha(175)

    arrow_pts = (
        (0, 100),
        (0, 200),
        (200, 200),
        (200, 300),
        (300, 150),
        (200, 0),
        (200, 100))

    speed = vel.length
    sps = [scale_seq(p, speed / 4, 1 / 4) for p in arrow_pts]

    ctr = centroid_seq(sps)
    cps = [translate_seq(p, -ctr[0], -ctr[1]) for p in sps]

    angle = np.arctan2(vel[1], vel[0])
    c = np.cos(angle)
    s = np.sin(angle)
    qs = [rotate_seq(p, c, s) for p in cps]

    ps = [translate_seq(p, loc[0], loc[1]) for p in qs]

    pygame.draw.polygon(
        arrow_surface,
        THECOLORS['white'],  # (0, 0, 0),
        ps,
        0)

    # ns = pygame.transform.rotate(arrow_surface, -angle)
    g_screen.blit(
        source=arrow_surface,
        dest=((0, 0)))  # ((loc - Vec2d(0, 150)).int_tuple))


def screen_cage(pad=10):
    return [Vec2d(pad, pad),
            Vec2d(pad, SCREEN_HEIGHT - pad - 1),
            Vec2d(SCREEN_WIDTH - pad - 1, SCREEN_HEIGHT - pad - 1),
            Vec2d(SCREEN_WIDTH - pad - 1, pad)]


def draw_cage():
    draw_int_tuples([p.int_tuple for p in screen_cage()], THECOLORS['green'])


def clear_screen(color=THECOLORS['black']):
    g_screen.fill(color)


#  ___
# |   \ ___ _ __  ___ ___
# | |) / -_) '  \/ _ (_-<
# |___/\___|_|_|_\___/__/


def demo_cage(pause=0.75, dt=1):
    me, them = mk_us()
    draw_us_with_arrows(me, them)

    draw_cage()

    # draw_perps_to_cage(me)
    # draw_perps_to_cage(them)

    cage = screen_cage()
    walls = pairwise_toroidal(cage, Wall)

    my_wall_prediction = wall_prediction(me, walls, dt)
    draw_vector(my_wall_prediction['puck_strike_point'],
                my_wall_prediction['wall_strike_point'],
                THECOLORS['purple1'])

    their_wall_prediction = wall_prediction(them, walls, dt)
    draw_vector(their_wall_prediction['puck_strike_point'],
                their_wall_prediction['wall_strike_point'],
                THECOLORS['purple1'])

    my_puck_prediction = me.predict_a_puck_collision(them, dt)

    # pp.pprint({'my_wall_prediction': my_wall_prediction,
    #            'their_wall_prediction': their_wall_prediction,
    #            'my_puck_prediction': my_puck_prediction,
    #            'their_puck_prediction': their_puck_prediction})

    nearest_wall_strike = arg_min([my_wall_prediction,
                                   their_wall_prediction],
                                  lambda p: p['tau'])

    # TODO: can't handle a double wall strike

    if my_puck_prediction['gonna_hit'] and \
            my_puck_prediction['tau'] < nearest_wall_strike['tau']:

        tau = my_puck_prediction['tau']
        step_and_draw_both(dt, me, tau, them)

        n = (them.center - me.center).normalized()

        perp = SCREEN_WIDTH * Vec2d(n[1], -n[0])

        strike = me.center + me.radius * n
        sanity = them.center - them.radius * n

        draw_vector(strike, sanity, THECOLORS['goldenrod1'])

        draw_vector(strike, strike + perp, THECOLORS['limegreen'])
        draw_vector(strike, strike - perp, THECOLORS['maroon1'])

        print({'puck strike': sanity - strike})

    else:  # strike the wall
        tau = nearest_wall_strike['tau']

        step_and_draw_both(dt, me, tau, them)

        print('wall strike')

    assert tau >= 0

    pygame.display.flip()

    time.sleep(pause)


def step_and_draw_both(dt, me, tau, them):
    me.step_many(int(tau), dt)
    them.step_many(int(tau), dt)
    me.draw()
    them.draw()


def wall_prediction(puck, walls, dt):
    predictions = \
        [puck.predict_a_wall_collision(wall, dt) for wall in walls]
    prediction = arg_min(
        predictions,
        lambda p: p['tau'] if p['tau'] >= 0 else np.inf)
    return prediction


def draw_us_with_arrows(me, them):
    me.draw()
    draw_centered_arrow(loc=me.center, vel=me.velocity)
    them.draw()
    draw_centered_arrow(loc=them.center, vel=them.velocity)


def mk_us():
    me = Puck(center=Vec2d(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2),
              velocity=random_velocity(),
              mass=100,
              radius=42,
              color=THECOLORS['red'])
    them = Puck(center=Vec2d(SCREEN_WIDTH / 1.5, SCREEN_HEIGHT / 2.5),
                velocity=random_velocity(),
                mass=100,
                radius=79,
                color=THECOLORS['green'])
    return me, them


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


def set_up_screen(pause=0.75):
    global g_screen
    pygame.init()
    g_screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    # clock = pygame.time.Clock()
    g_screen.set_alpha(None)
    time.sleep(pause)


def random_velocity():
    # speed = np.random.randint(1, 5)
    speed = 1 + 4 * np.random.rand()
    # direction = Vec2d(1, 0).rotated(np.random.randint(-2, 2))
    direction = Vec2d(1, 0).rotated(4 * np.random.rand() - 2)
    result = speed * direction
    return result


def demo_hull(pause=7.0):
    clear_screen()
    puck = Puck(
        center=Vec2d(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2),
        velocity=Vec2d(1, 1 / 2),
        mass=100)
    puck.step_many(DEMO_STEPS, DEMO_DT)
    hull = convex_hull(random_points(15))
    draw_int_tuples(hull)
    pairwise_toroidal(hull,
                      lambda p0, p1: draw_collinear_point_and_param(
                          Vec2d(p0), Vec2d(p1), line_color=THECOLORS['purple']))
    time.sleep(pause)


#   ___ _            _       ___     _ _ _    _
#  / __| |__ _ _____(_)__   / __|___| | (_)__(_)___ _ _  ___
# | (__| / _` (_-<_-< / _| | (__/ _ \ | | (_-< / _ \ ' \(_-<
#  \___|_\__,_/__/__/_\__|  \___\___/_|_|_/__/_\___/_||_/__/


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
        self.space.step(1. / 10)
        pygame.display.flip()


def demo_classic(steps=500):
    game_state = GameState()
    for _ in range(steps):
        game_state.frame_step()


#  __  __      _
# |  \/  |__ _(_)_ _
# | |\/| / _` | | ' \
# |_|  |_\__,_|_|_||_|


def main():
    global g_screen
    set_up_screen()
    # demo_hull(0.75)
    for _ in range(20):
        demo_cage(pause=2, dt=0.001)
        clear_screen()
    # demo_classic(steps=3000)
    # input('Press [Enter] to end the program.')


if __name__ == "__main__":
    main()
