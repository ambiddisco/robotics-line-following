import numpy as np
import roboticstoolbox as rtb
import spatialgeometry as sg
import spatialmath as sm
import swift
import time
import matplotlib as mtb
import matplotlib.pyplot as plt

from typing import Sequence, Tuple, List, Dict


# Define colour variables, for simulation environment

# --- SIMULATION CONSTANTS & COLORS ---
FLOOR_HEIGHT = 0.02
COLOR_BLUE = (0.2, 0.5, 0.8, 1.0)   # Robot color
COLOR_WALL = (0.3, 0.3, 0.3, 1.0)   # Wall color
COLOR_RAY = (1.0, 0.8, 0.0, 0.7)    # Sensor ray color (Yellow)
COLOR_GREEN = (0.2, 0.8, 0.3, 0.7)  # Waypoint/Target color
COLOR_GREY = (0.5, 0.5, 0.5, 0.8)   # Path line color
COLOR_WHITE = (1.0, 1.0, 1.0, 0.6)  # World ground


# environment setup

env = swift.Swift()
env.launch(realtime=True,comms="rtc",browser="browser")
#env.set_camera_pose([0,0,0])

# Simulation parameters
dt = 0.01  # time step
# time = np.arange(0, 10, dt)  # simulation time

# world bounds

world_bounds = (-2.5, 2.5, -2.5, 2.5)  # min_x, max_x, min_y, max_y

# so that the browser does not crash
time.sleep(1)

def add_world_bounds(env: swift.Swift, bounds: Tuple[float, float, float, float]) -> sg.Cuboid:
    """Draw a translucent floor tile highlighting the workspace."""
    min_x, max_x, min_y, max_y = bounds
    width, depth = max_x - min_x, max_y - min_y
    floor = sg.Cuboid(scale=[width, depth, FLOOR_HEIGHT], color=COLOR_WHITE)
    floor.T = sm.SE3((min_x + max_x) / 2.0, (min_y + max_y) / 2.0, 0.01)
    env.add(floor)
    return floor

add_world_bounds(env, world_bounds)


COLOR_RED = (0.8, 0.2, 0.3, 1.0)



def select_point_on_edge(edge_num, bounds):
    point = np.random.uniform(bounds[0], bounds[1])
    if 0 <= edge_num <= 1:
        return bounds[edge_num], point
    return point, bounds[edge_num]

def create_random_line(e, bounds):
    # line: ax+b
    # must pass through env
    # will be defined by points inside env
    random_edge_1 = np.random.randint(0,4)
    random_edge_2 = np.random.randint(0,4)
    while random_edge_1 == random_edge_2:
        random_edge_2 = np.random.randint(0,4)
    # select 2 separate edges to connect

    x_1, y_1 = select_point_on_edge(random_edge_1, bounds)
    x_2, y_2 = select_point_on_edge(random_edge_2, bounds)

    dx, dy = x_1 - x_2, y_1 - y_2
    length = np.hypot(dx, dy)
    angle = np.arctan2(dy, dx)
    cx, cy = (x_1 + x_2) / 2.0, (y_1 + y_2) / 2.0

    line = sg.Cuboid(scale=[length, 0.03, 0.01], color=COLOR_RED)
    line.T = sm.SE3(cx, cy, 0.015) * sm.SE3.Rz(angle)
    env.add(line)

create_random_line(env, world_bounds)


# so that the browser does not crash
time.sleep(1)



def select_random_target(e, bounds):
    x = np.random.uniform(bounds[0], bounds[1])
    y = np.random.uniform(bounds[2], bounds[3])
    z = 0.15 # so not touching ground

    return x, y, z

def add_random_objects(e, bounds, obj_num):
    obj_positions = []

    for obj in range(obj_num):
        x, y, z = select_random_target(e, bounds)
        
        # maybe for future: make objects so they cannot overlap
        # otherwise you have simply created a larger object

        # while (x,y,z) in obj_positions:
        #     x, y, z = select_random_target(e, bounds)
        
        cube_size = np.random.uniform(0.25, 1) # later do this
        
        block = sg.Cuboid(scale=[cube_size] * 3, color=COLOR_BLUE) 
        block.T = sm.SE3(x, y, z)
        
        env.add(block)
    
add_random_objects(env, world_bounds, 3)


