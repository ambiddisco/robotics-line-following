# Line follower project
## Robot Control Applications
#### Group Biddiscombe, Martinez, Spanò

# IMPORTS -----------------------------------------------------
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence, Tuple, List, Dict
import numpy as np
import roboticstoolbox as rtb
import spatialgeometry as sg
import spatialmath as sm
import swift
import time
import math
import matplotlib.pyplot as plt
import matplotlib as mtb

from unicycle_controller import TrajectoryTracker, HybridNavigator

# Set default matplotlib params
mtb.rcParams.update(mtb.rcParamsDefault)

# Global default values
FLOOR_HEIGHT = 0.02
WORLD_BOUNDS = [0., 10., 0., 10.]

COLOUR_BLUE = (0.2, 0.5, 0.8, 1.0)   # Robot colour
COLOUR_WALL = (1.0, 0.2, 0.2, 1.0)   # Obstacle colour
COLOUR_RAY = (1.0, 0.8, 0.0, 0.7)    # Sensor ray colour (Yellow)
COLOUR_GREEN = (0.2, 0.8, 0.3, 0.7)  # Waypoint/Target colour
COLOUR_PATH = (0.1, 0.6, 0.3, 0.8)   # Path line colour
COLOUR_WHITE = (1.0, 1.0, 1.0, 0.6)  # World ground

# SETUP FUNCTIONS -----------------------------------------------------
# functions for the environment creation and setup

def add_floor(e: swift.Swift, world_bounds = WORLD_BOUNDS, floor_height: float = FLOOR_HEIGHT) -> sg.Cuboid:
    """Adds a visual marker for world bounds."""
    min_x, max_x, min_y, max_y = world_bounds
    width, depth = max_x - min_x, max_y - min_y
    floor = sg.Cuboid(scale=[width, depth, FLOOR_HEIGHT], color=COLOUR_WHITE)
    floor.T = sm.SE3((min_x + max_x) / 2.0, (min_y + max_y) / 2.0, floor_height)
    e.add(floor)
    return floor

def create_path(world_bounds = WORLD_BOUNDS, type: str = "line") -> List[Tuple[float, float]]: 
    """Returns a list of waypoints the path to be followed passes through."""
    min_x, max_x, min_y, max_y = world_bounds

    if type == "line":
        # Random line: y = a x + b, stay in positive area
        a = np.random.uniform(0.5, 2.)
        b = 0 # np.random.uniform(0., 2.)

        max_x_line = min(max_x, (max_y - b) / a)
        num_waypoints = 50
        xs = np.linspace(min_x, max_x_line, num_waypoints)
        waypoints = [(float(x), float(a * x + b)) for x in xs]

        print(f"[ENV] Path constant line:    y = {a:.3f} x + {b:.3f}")
    
    if type == "sine":
        a = np.random.uniform(0.5, 2.)
        b = (min_y + max_y) / 2 # always centred
        c = np.random.uniform(-1, 1)

        num_waypoints = 70
        xs = xs = np.linspace(min_x, max_x, num_waypoints)
        waypoints = [(float(x), float(np.sin(x * a + c) + b)) for x in xs]
        print(f"[ENV] Path sine wave:    y = sin({a:.3f} x + {c:.3f}) + {b:.3f}")

    print(f"[ENV] Path start state:  x = {waypoints[0][0]}, y = {waypoints[0][1]}")
    print(f"[ENV] Path end state:    x = {waypoints[-1][0]}, y = {waypoints[-1][1]}")
    return waypoints

def add_path_visual(e: swift.Swift, waypoints: List[Tuple[float, float]], floor_height: float = FLOOR_HEIGHT) -> None:
    """Draws green(ish) static lines connecting the waypoints."""
    for i in range(len(waypoints) - 1):
        p1, p2 = waypoints[i], waypoints[i+1]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = np.hypot(dx, dy)
        angle = np.arctan2(dy, dx)
        cx, cy = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
        
        line = sg.Cuboid(scale=[length, 0.03, 0.01], color=COLOUR_PATH)
        line.T = sm.SE3(cx, cy, floor_height + 0.01) * sm.SE3.Rz(angle)
        e.add(line)

def get_random_coords_in_bounds(world_bounds = WORLD_BOUNDS, floor_height: float = FLOOR_HEIGHT) -> Tuple[float, float, float]:
    """Method for returning non-edge coordinates inside valid borders."""
    x = np.random.uniform(world_bounds[0] + 1, world_bounds[1] - 1)
    y = np.random.uniform(world_bounds[2] + 1, world_bounds[3] - 1)
    return (x, y, floor_height)

def get_position_on_line(waypoints: List[Tuple[float, float]], noise: float = 0.5, padding: float = 0.2, floor_height = FLOOR_HEIGHT) -> Tuple[float, float, float]:
    """Return coordinates near path, to be used with object placement."""
    # ignore first 20% and last 20% of line (default padding is 0.2)
    pos_on_line = np.random.randint(int(padding * len(waypoints)), int((1 - padding) * len(waypoints)))
    pos_on_line = waypoints[pos_on_line]
    x_noise = np.random.uniform(-noise, noise)
    y_noise = np.random.uniform(-noise, noise)
    return (pos_on_line[0] + x_noise, pos_on_line[1] + y_noise, floor_height)
    
def add_block(e: swift.Swift, cx: float, cy: float, w: float, l: float, floor_height: float = FLOOR_HEIGHT) -> sg.Cuboid:
    """Helper function to add a static block to the environment."""
    block = sg.Cuboid(scale=[w, l, 0.5], color=COLOUR_WALL)
    block.T = sm.SE3(cx, cy, floor_height + 0.25)
    e.add(block)
    return block

def add_cylinder(e: swift.Swift, cx: float, cy: float, r: float, floor_height: float = FLOOR_HEIGHT) -> sg.Cylinder:
    """Helper function to add a static cylinder to the environment."""
    cylinder = sg.Cylinder(radius=r, length=0.5, color=COLOUR_WALL)
    cylinder.T = sm.SE3(cx, cy, floor_height + 0.25)
    e.add(cylinder)
    return cylinder

def create_obstacles(e: swift.Swift, waypoints, num_obstacles: int = 6, mixed_objects: bool = False, spacing: float = 1., world_bounds = WORLD_BOUNDS):
    """Adds obstacles to enviroment, both near (or on) the path and randomly in the world bounds."""
    current_positions = []
    obstacles = []

    while len(current_positions) < num_obstacles:
        prob = np.random.uniform(0,1)
        if prob > 0.8:
            pos = get_random_coords_in_bounds()
        elif prob > 0.5:
            pos = get_position_on_line(waypoints, noise = 1.5)
        else:
            pos = get_position_on_line(waypoints, noise = 0.5)
        
        for item in current_positions:
            d = np.hypot(item[0] - pos[0], item[1] - pos[1])
            if d < spacing: 
                # space them a little apart, if too close discard and repick a location
                continue
        current_positions.append(pos)
        
        if mixed_objects:
            prob = np.random.uniform(0,1)
            if prob > 0.5:
                radius = float(np.random.uniform(0.4, 0.7))
                obstacles.append(add_cylinder(e, pos[0], pos[1], radius))
                continue # skips the rest if a cylinder has been added, else adds a block

        width = float(np.random.uniform(0.4, 2.))
        length = float(np.random.uniform(0.4, 2.))

        obstacles.append(add_block(e, pos[0], pos[1], width, length))

    print(f"[ENV] Created {len(obstacles)} obstacles.")
    return obstacles

# ROBOT -----------------------------------------------------
# Uses environment so can't move to separate file

@dataclass
class AgentVisual:
    """
    A wrapper class for the 3D visual object (Cuboid) representing the robot.
    It updates the position of the 3D block in the Swift browser simulator.
    """
    body: sg.Cuboid

    def update(self, base_pose: sm.SE3) -> None:
        """Moves the visual block to the robot's new physical pose."""
        # Offset by FLOOR_HEIGHT so the robot sits on top of the floor, not inside it
        self.body.T = base_pose * sm.SE3(0.0, 0.0, 0.05 + FLOOR_HEIGHT)

def make_agent(c):
    """Factory function to create the visual block (chassis)."""
    return AgentVisual(sg.Cuboid(scale=[0.4, 0.2, 0.1], color=c))

def state_to_pose(s):
    """Converts state vector [x, y, theta] into an SE3 Homogeneous Transformation Matrix."""
    return sm.SE3(s[0], s[1], 0.0) * sm.SE3.Rz(s[2])

class UnicycleAgent:
    """
    Combines Physics (roboticstoolbox) and Graphics (spatialgeometry).
    acts as the 'Digital Twin' of the robot.
    """
    def __init__(self, n, s, c) -> None:
        self.name = n
        # The kinematic model (handles the math of how wheels move the robot)
        self._robot = rtb.Unicycle()
        self._robot.init(np.array(s, dtype=float)) # Set initial state
        self._visual = make_agent(c)

    def register(self, e: swift.Swift) -> None:
        """Adds the robot visual to the Swift environment."""
        e.add(self._visual.body)
        self.update_visual()

    def state(self) -> np.ndarray:
        """Returns current [x, y, theta]."""
        return np.asarray(self._robot.q, dtype=float)

    def update_visual(self) -> None:
        """Syncs the visual block with the internal mathematical state."""
        self._visual.update(state_to_pose(self.state()))

    def apply_velocity_cmd(self, c) -> None:
        """
        Integrates the physics.
        Args: c = [v, omega] (Linear and Angular velocity)
        """
        self._robot.step(np.array(c, dtype=float)) # Physics step
        self.update_visual()                       # Graphics step

# ROBOT SENSORS -----------------------------------------------------

@dataclass
class SensorSim: # NOTE: Ask Riccardo about this bit
    """
    Simulates LIDAR/Ultrasonic sensors using Raycasting.
    Calculates the distance from the robot center to the nearest wall 
    along specific angles.
    """
    walls: List[sg.Cuboid]
    max_range: float = 5.0
    # Define sensor directions relative to robot heading (0 = Front)
    sensor_angles_rad: Dict[str, float] = field(default_factory=lambda: {
        'front': 0.0, 
        'front_left': np.pi/4,   # +45 degrees
        'front_right': -np.pi/4  # -45 degrees
    })
    
    def _ray_aabb_intersection(self, ro, rd, wall) -> float:
        """
        Calculates intersection between a Ray and an Axis-Aligned Bounding Box (Wall).
        standard algorithm in computer graphics for collision detection.
        
        Args:
            ro: Ray Origin [x, y]
            rd: Ray Direction vector [dx, dy]
            wall: The wall object to check against
        Returns:
            Distance to intersection, or max_range if no hit.
        """
        # Extract wall position (c) and size (s)
        c = wall.T[0:3, 3]
        s = wall.scale
        # Calculate Min (mb) and Max (Mb) corners of the wall
        mb, Mb = c - s / 2.0, c + s / 2.0
        
        t_min, t_max = 0.0, self.max_range
        
        # Check intersection for X and Y axes
        for i in range(2):
            if abs(rd[i]) < 1e-6: # Ray is parallel to the axis
                if ro[i] < mb[i] or ro[i] > Mb[i]: return self.max_range
            else:
                # Slab method for AABB intersection
                t1 = (mb[i] - ro[i]) / rd[i]
                t2 = (Mb[i] - ro[i]) / rd[i]
                t_n = min(t1, t2)
                t_f = max(t1, t2)
                t_min = max(t_min, t_n)
                t_max = min(t_max, t_f)
                
                if t_min > t_max or t_max < 0: return self.max_range
                
        return min(t_min, self.max_range) if t_min > 0 else self.max_range

    def get_readings(self, s: Sequence[float]) -> Dict[str, float]:
        """
        Main public method. Returns the distance readings for all sensors.
        Args: s = Robot state [x, y, theta]
        """
        x, y, theta = s
        ro = np.array([x, y]) # Ray Origin
        readings = {}
        
        for name, angle in self.sensor_angles_rad.items():
            # Global Angle of the sensor = Robot Heading + Sensor Mount Angle
            ga = theta + angle
            rd = np.array([math.cos(ga), math.sin(ga)]) # Ray Direction Unit Vector
            
            min_dist = self.max_range
            # Check against every wall and find the closest one
            for wall in self.walls: 
                dist = self._ray_aabb_intersection(ro, rd, wall)
                min_dist = min(min_dist, dist)
                
            readings[name] = min_dist
        return readings
    
# VISUAL PART OF RAYS -----------------------------------------------------
def make_ray_visual(c): 
    """Creates a thin yellow stick to represent a laser ray."""
    return sg.Cuboid(scale=[1.0, 0.02, 0.02], color=c)

def update_ray_visuals(rp, r, sa, vr):
    """Updates the length and rotation of the yellow sticks based on sensor data."""
    for name, ray_visual in vr.items():
        dist = r.get(name, 0.0)
        angle = sa.get(name, 0.0)
        ray_visual.scale[0] = dist # Stretch the stick to match distance
        # Complex transform: Robot Pose * Sensor Rotation * Shift stick center forward
        ray_visual.T = rp * sm.SE3.Rz(angle) * sm.SE3(dist / 2.0, 0, 0.05)

# START CREATING THE PARTS TO BE USED
# NOTE: create all objects before swift environment is started, then place them visually
# because it has been problematic in the past, avoid stressing after opening swift env 

# CREATE MAIN -----------------------------------------------------
def main():
    # Create the path, can be of type "line" for a straight path or "sine" for a curved path
    WAYPOINTS = create_path(type="sine") # "line" or "sine"

    start_state = [WAYPOINTS[0][0], WAYPOINTS[0][1], 0] # Start at (0, -4), facing UP (pi/2)

    # CREATE AGENTS -----------------------------------------------------

    robot_agent = UnicycleAgent("robot", start_state, COLOUR_BLUE)

    # Initialize the HYBRID CONTROLLER
    controller = HybridNavigator(
        # Planner Params (Go-to-point behavior)
        waypoints=WAYPOINTS, 
        reference_speed=0.8,
        linear_velocity_gain=2.0, 
        angular_velocity_gain=4.0,
        
        # Guard Params (Wall Following behavior)
        reactive_angular_gain=2.5, # How aggressively to turn away from wall
        safety_distance=1.0,       # Trigger distance (panic threshold)
        wall_follow_distance=0.5,  # Target distance to keep from wall
        reactive_speed=0.4,        # Slower speed when near walls
        
        # Physical Limits
        max_linear_velocity=1.0,
    )

    # Sensor for robot needs to be added after the walls, so goes after opening swift env

    # ENVIRONMENT SETUP -----------------------------------------------------

    # Simulation timing settings
    dt = 0.05
    simulation_time = 40.0
    simulation_steps = int(simulation_time / dt)

    # Launch Swift
    env = swift.Swift()
    env.launch(realtime=True, comms="rtc", browser="browser")

    time.sleep(1.0) # added so it breaks less oten

    # Set the camera pose
    env.set_camera_pose(position=[5.5, 1.0, 7.0], look_at=[5.5, 3., 2.0])
    # env.step(0)

    # Add the world visuals:

    # Add the floor, defines world borders
    FLOOR = add_floor(env)

    # Add the path visually to the enviroment
    add_path_visual(env, WAYPOINTS)

    # Add obstacles, can be just blocks or of mixed block and cylinder type
    walls = create_obstacles(env, waypoints=WAYPOINTS, mixed_objects=False, num_obstacles=5)

    # Lock all the changes into the environment by performing a step
    # env.step(0)

    # Create the Sensor System (Math)
    # Depends on walls
    sensor_sim = SensorSim(walls=walls, max_range=3.0)

    # Create yellow rays for visualization
    viz_rays = {name: make_ray_visual(COLOUR_RAY) for name in sensor_sim.sensor_angles_rad}
    # Create green sphere for the "Virtual Rabbit" (Trajectory Target)
    ref_marker = sg.Sphere(radius=0.08, color=COLOUR_GREEN)

    # Add everything to the scene
    robot_agent.register(env)
    add_path_visual(env, WAYPOINTS)
    env.add(ref_marker)
    for ray_viz in viz_rays.values(): env.add(ray_viz)

    # Render initial frame
    env.step(0.0) 
    time.sleep(1.0) # added so it breaks less oten

    # Data Logging setup
    time_log, x_log, y_log = [], [], []
    v_cmd_log, w_cmd_log = [], []
    tracking_error_log = []

    target_arrived = False

    # MAIN SIMULATION LOOP -----------------------------------------------------

    for step in range(simulation_steps):
        t = step * dt
        r_state = robot_agent.state() # Get current [x, y, theta]
        
        # A. SENSE: Get distances to walls
        readings = sensor_sim.get_readings(r_state)

        # B. PLAN & DECIDE: Compute Control Inputs
        # The controller internally decides whether to follow the path or the wall
        v_cmd, w_cmd = controller.compute_control(
            r_state, t=t,
            d_front=readings['front'], 
            d_front_left=readings['front_left'], 
            d_front_right=readings['front_right']
        )

        # C. ACT: Apply velocities to the robot
        robot_agent.apply_velocity_cmd([v_cmd, w_cmd])
        
        # Update Visuals (Yellow rays and Green Target)
        update_ray_visuals(state_to_pose(r_state), readings, sensor_sim.sensor_angles_rad, viz_rays)
        
        ref_x, ref_y = controller._compute_reference_point(t)
        ref_marker.T = sm.SE3(ref_x, ref_y, 0.01)
        
        # D. LOG DATA
        time_log.append(t); x_log.append(r_state[0]); y_log.append(r_state[1])
        v_cmd_log.append(v_cmd); w_cmd_log.append(w_cmd)
        # Error = Distance between Robot and Green Marker
        tracking_error_log.append(np.hypot(ref_x - r_state[0], ref_y - r_state[1]))

        # when controller reaches endpoint of target line    
        if controller.is_done():
            if not target_arrived:
                print(f"Target arrived at t={t:.2f}s.")
                target_arrived = True
            if tracking_error_log[-1] < 0.05:
                print(f"Path completed at t={t:.2f}s.")
                break;
            
        # Advance the simulator clock
        env.step(dt)

    env.close()

    # PLOTTING -----------------------------------------------------

    print("Simulation ended. Generating plots...")

    average_error = np.mean(tracking_error_log)

    time_log = np.array(time_log)
    fig, axs = plt.subplots(3, 1, figsize=(10, 18))
    fig.suptitle("Hybrid Navigation Analysis", fontsize=16)

    # Plot 1: XY Trajectory (Top view of the maze)
    axs[0].plot(x_log, y_log, "b-", label="Actual Robot Path")
    wp_x = [p[0] for p in WAYPOINTS]; wp_y = [p[1] for p in WAYPOINTS]
    axs[0].plot(wp_x, wp_y, "g--", linewidth=2, label="Planner Trajectory (Desired)")
    axs[0].plot(start_state[0], start_state[1], "go", markersize=10, label="Start")

    # Draw walls on the plot
    for wall in walls: 
        c = wall.T[0:3, 3]; s = wall.scale
        xm, xM = c[0]-s[0]/2, c[0]+s[0]/2
        ym, yM = c[1]-s[1]/2, c[1]+s[1]/2
        axs[0].fill([xm, xM, xM, xm], [ym, ym, yM, yM], color='r', alpha=0.6, label="Walls")

    # Handle legend deduplication
    handles, labels = axs[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs[0].legend(by_label.values(), by_label.keys(), loc='upper left')

    axs[0].set_xlabel("X [m]"); axs[0].set_ylabel("Y [m]"); axs[0].set_title("Robot Path vs Plan")
    axs[0].grid(True); axs[0].axis("equal")

    # Plot 2: Control Inputs (v, w) over time
    axs[1].plot(time_log, v_cmd_log, "b", label="Lin. Velocity (v)")
    axs[1].axhline(y=controller._v_ref, color='c', linestyle='--', label="Ref Speed")
    axs[1].set_ylabel("v [m/s]"); axs[1].legend(loc="upper left"); axs[1].grid(True)

    ax2 = axs[1].twinx() # Secondary Y-axis for angular velocity
    ax2.plot(time_log, w_cmd_log, "m", label="Ang. Velocity (w)")
    ax2.set_ylabel("w [rad/s]"); ax2.legend(loc="upper right")
    axs[1].set_title("Control Inputs")

    # Plot 3: Tracking Error
    axs[2].plot(time_log, tracking_error_log, "r-", label="Tracking Error")
    axs[2].set_xlabel("Time [s]"); axs[2].set_ylabel("Error [m]"); axs[2].grid(True)
    axs[2].axhline(average_error, color="orange", linestyle="--", label="Average error")
    axs[2].set_title("Tracking Error (Peaks indicate Obstacle Avoidance active)")
    axs[2].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()

    plt.savefig("hybrid_navigation_results.png", dpi=150)
    print("Plots saved to hybrid_navigation_results.png")


if __name__ == "__main__":
    main()