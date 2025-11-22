"""
UNICYCLE CONTROLLERS LIBRARY
----------------------------
This module implements a hierarchy of control algorithms for a unicycle mobile robot.
It evolves from basic position control to advanced hybrid navigation.

Classes:
    - UnicycleController: Base class with shared math and saturation logic.
    - Wanderer: Random-walk behavior inside a workspace.
    - Pursuer: Drives directly toward a moving target.
    - LineFollower: Basic implementation for Project #4 (Constant speed, Line tracking).
    - WaypointFollower: Sequential waypoint navigation with constant speed.
    - ReactiveNavigator: Potential Fields navigation (Waypoints + Obstacle Repulsion).
    - TrajectoryTracker: Advanced extension (Time-based tracking of a moving reference).
    - CorridorFollower: Purely reactive navigation using simulated distance sensors.
    - HybridNavigator: The final solution fusing Trajectory Tracking and Reactive safety.

Author: Alexandra Maria Biddiscombe, Carlos Martinez Gil, Riccardo Angelo Spanò
Course: Robot Control Applications
"""

from typing import Optional, Sequence, Tuple, List, Dict
import numpy as np
import math 

# ==============================================================================
# --- BASE CLASS ---
# ==============================================================================

def _wrap_angle(angle: float) -> float:
    """
    Wrap an angle to the [-pi, pi] interval.
    Essential for calculating the smallest difference between two headings.
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

class UnicycleController:
    """
    Base class containing shared logic for unicycle kinematic models.
    Implements velocity limits (saturation) and a proportional 'go-to-point' control law.
    """

    def __init__(
        self,
        *,
        linear_velocity_gain: float = 0.5,   # k_v
        angular_velocity_gain: float = 1.0,  # k_h
        max_linear_velocity: float = 0.5,
        max_angular_velocity: float = 1.5,
    ) -> None:
        self._k_v = float(linear_velocity_gain)
        self._k_h = float(angular_velocity_gain)
        self._v_max = float(max_linear_velocity)
        self._w_max = float(max_angular_velocity)

    @property
    def max_linear_velocity(self) -> float:
        return self._v_max

    @property
    def max_angular_velocity(self) -> float:
        return self._w_max

    def set_max_velocities(self, linear: float, angular: float) -> None:
        self._v_max = float(linear)
        self._w_max = float(angular)

    def set_gains(self, k_v: float, k_h: float) -> None:
        self._k_v = float(k_v)
        self._k_h = float(k_h)

    def _clip(self, v: float, omega: float) -> Tuple[float, float]:
        """Clip controller outputs to configured limits."""
        v_c = float(np.clip(v, -self._v_max, self._v_max))
        w_c = float(np.clip(omega, -self._w_max, self._w_max))
        return v_c, w_c

    def go_to_point(self, current_state: Sequence[float], goal_xy: Tuple[float, float]) -> Tuple[float, float]:
        """
        Drive the unicycle toward a 2D goal using the simple proportional law.
        v = k_v * error_distance
        w = k_h * error_heading
        """
        x, y, theta = float(current_state[0]), float(current_state[1]), float(current_state[2])
        gx, gy = float(goal_xy[0]), float(goal_xy[1])

        dx = gx - x
        dy = gy - y
        rho = float(np.hypot(dx, dy))    # distance to goal
        theta_star = float(np.arctan2(dy, dx)) # desired heading to goal
        heading_err = _wrap_angle(theta_star - theta)

        v = self._k_v * rho
        omega = self._k_h * heading_err
        return self._clip(v, omega)

    def compute_control(self, state: Sequence[float], *args, **kwargs) -> Tuple[float, float]:  # pragma: no cover
        """Interface method to be implemented by subclasses."""
        raise NotImplementedError


# ==============================================================================
# --- EXAMPLES: CAT_AND_MOUSE ---
# ==============================================================================

class Wanderer(UnicycleController):
    """Random-walk behavior inside a square workspace using the simple controller."""

    def __init__(
        self,
        *,
        linear_velocity_gain: float = 0.6,
        angular_velocity_gain: float = 3.0,
        max_linear_velocity: float = 0.5,
        max_angular_velocity: float = 1.5,
        world_bounds: Tuple[float, float, float, float] = (-2.0, 2.0, -2.0, 2.0),  # min_x, max_x, min_y, max_y
        goal_tolerance: float = 0.15,  # meters
    ) -> None:
        super().__init__(
            linear_velocity_gain=linear_velocity_gain,
            angular_velocity_gain=angular_velocity_gain,
            max_linear_velocity=max_linear_velocity,
            max_angular_velocity=max_angular_velocity,
        )
        self._bounds = world_bounds
        self._goal_tol = float(goal_tolerance)
        self._rng = np.random.default_rng()
        self._goal: Optional[Tuple[float, float]] = None  # (x*, y*)

    def compute_control(self, state: Sequence[float], *_, **__) -> Tuple[float, float]:
        x = float(state[0])
        y = float(state[1])

        if self._goal is None or np.hypot(self._goal[0] - x, self._goal[1] - y) <= self._goal_tol:
            self._goal = self._sample_goal_xy()

        return self.go_to_point(state, self._goal)

    def _sample_goal_xy(self) -> Tuple[float, float]:
        min_x, max_x, min_y, max_y = self._bounds
        x = float(self._rng.uniform(min_x, max_x))
        y = float(self._rng.uniform(min_y, max_y))
        return x, y

    def reset_goal(self) -> None:
        self._goal = None

    def sample_pose(self) -> Tuple[float, float, float]:
        x, y = self._sample_goal_xy()
        theta = float(self._rng.uniform(-np.pi, np.pi))
        return x, y, theta


class Pursuer(UnicycleController):
    """Drive directly toward a moving target's current position."""

    def __init__(
        self,
        *,
        linear_velocity_gain: float = 0.9,
        angular_velocity_gain: float = 3.5,
        max_linear_velocity: float = 0.6,
        max_angular_velocity: float = 2.5,
        capture_threshold: float = 0.25,
    ) -> None:
        super().__init__(
            linear_velocity_gain=linear_velocity_gain,
            angular_velocity_gain=angular_velocity_gain,
            max_linear_velocity=max_linear_velocity,
            max_angular_velocity=max_angular_velocity,
        )
        self._capture_thresh = float(capture_threshold)

    def compute_control(self, state: Sequence[float], target_state: Sequence[float], *_, **__) -> Tuple[float, float]:
        if self.has_captured(state, target_state):
            return 0.0, 0.0
        tx, ty = float(target_state[0]), float(target_state[1])
        return self.go_to_point(state, (tx, ty))

    def has_captured(self, state: Sequence[float], target_state: Sequence[float]) -> bool:
        sx, sy = float(state[0]), float(state[1])
        tx, ty = float(target_state[0]), float(target_state[1])
        return np.hypot(tx - sx, ty - sy) <= self._capture_thresh

    def reset(self) -> None:
        return


# ==============================================================================
# --- PROJECT 4: LINE FOLLOWER (BASE) ---
# ==============================================================================

class LineFollower(UnicycleController):
    """Constant velocity control to follow a horizontal line y = y_ref."""

    def __init__(
        self,
        *,
        angular_velocity_gain: float = 3.0,  # k_h
        max_linear_velocity: float = 0.5,
        max_angular_velocity: float = 2.5,
        const_linear_velocity: float = 0.5,  # v0
        lookahead_distance: float = 0.4,     # L
        line_y_ref: float = 0.0,
    ) -> None:
        super().__init__(
            linear_velocity_gain=0.0,  # Not used for speed control here
            angular_velocity_gain=angular_velocity_gain,
            max_linear_velocity=max_linear_velocity,
            max_angular_velocity=max_angular_velocity,
        )
        self._v0 = float(const_linear_velocity)
        self._L = float(lookahead_distance)
        self._y_ref = float(line_y_ref)

    def compute_control(self, state: Sequence[float], *_, **__) -> Tuple[float, float]:
        x, y, theta = float(state[0]), float(state[1]), float(state[2])
        v = self._v0
        
        # Carrot-chasing logic (Pure Pursuit variation)
        x_goal, y_goal = x + self._L, self._y_ref
        dx, dy = x_goal - x, y_goal - y
        theta_star = float(np.arctan2(dy, dx))
        
        heading_err = _wrap_angle(theta_star - theta)
        omega = self._k_h * heading_err
        return self._clip(v, omega)

    def get_errors(self, state: Sequence[float]) -> Tuple[float, float]:
        """Helper for debugging."""
        x, y, theta = float(state[0]), float(state[1]), float(state[2])
        dist_error = y - self._y_ref
        
        x_goal, y_goal = x + self._L, self._y_ref
        dx, dy = x_goal - x, y_goal - y
        theta_star = float(np.arctan2(dy, dx))
        heading_err = _wrap_angle(theta_star - theta)
        return dist_error, heading_err


# ==============================================================================
# --- EXTENSION 1: WAYPOINT FOLLOWER ---
# ==============================================================================

class WaypointFollower(UnicycleController):
    """Constant velocity control to follow a sequence of waypoints."""

    def __init__(
        self,
        *,
        angular_velocity_gain: float = 3.0,  # k_h
        max_linear_velocity: float = 0.5,
        max_angular_velocity: float = 2.5,
        const_linear_velocity: float = 0.5,  # v0
        waypoints: Sequence[Tuple[float, float]] = [(1.0, 0.0)],
        goal_tolerance: float = 0.1,  # [m]
    ) -> None:
        super().__init__(
            linear_velocity_gain=0.0,  # Not used
            angular_velocity_gain=angular_velocity_gain,
            max_linear_velocity=max_linear_velocity,
            max_angular_velocity=max_angular_velocity,
        )
        self._v0 = float(const_linear_velocity)
        self._waypoints = [tuple(wp) for wp in waypoints]
        self._goal_tol = float(goal_tolerance)
        self._target_idx = 0
        self._is_done = False

    def is_done(self) -> bool:
        return self._is_done

    def get_current_goal(self) -> Optional[Tuple[float, float]]:
        if self._is_done: return None
        return self._waypoints[self._target_idx]

    def _check_goal_arrival(self, x: float, y: float) -> None:
        if self._is_done: return
        gx, gy = self._waypoints[self._target_idx]
        dist_to_goal = np.hypot(gx - x, gy - y)

        if dist_to_goal <= self._goal_tol:
            print(f"[WaypointFollower] Reached waypoint {self._target_idx}: ({gx:.2f}, {gy:.2f})")
            if self._target_idx < len(self._waypoints) - 1:
                self._target_idx += 1
            else:
                self._is_done = True
                print("[WaypointFollower] Path complete.")

    def compute_control(self, state: Sequence[float], *_, **__) -> Tuple[float, float]:
        x, y, theta = float(state[0]), float(state[1]), float(state[2])
        self._check_goal_arrival(x, y)
        if self._is_done:
            return 0.0, 0.0
        
        gx, gy = self._waypoints[self._target_idx]
        v = self._v0
        
        dx, dy = gx - x, gy - y
        theta_star = float(np.arctan2(dy, dx))
        heading_err = _wrap_angle(theta_star - theta)
        omega = self._k_h * heading_err
        return self._clip(v, omega)


# ==============================================================================
# --- EXTENSION 2: REACTIVE NAVIGATOR (Potential Fields) ---
# ==============================================================================

class ReactiveNavigator(UnicycleController):
    """Follows waypoints AND avoids a moving obstacle using Repulsive Fields."""

    def __init__(
        self,
        *,
        angular_velocity_gain: float = 3.0,
        repulsion_gain: float = 0.5,
        waypoints: Sequence[Tuple[float, float]] = [(1.0, 0.0)],
        goal_tolerance: float = 0.15,
        const_linear_velocity: float = 0.5,
        detection_radius: float = 1.0,
        min_linear_velocity: float = 0.1,
        max_linear_velocity: float = 0.5,
        max_angular_velocity: float = 2.5,
    ) -> None:
        super().__init__(
            linear_velocity_gain=0.0,
            angular_velocity_gain=angular_velocity_gain,
            max_linear_velocity=max_linear_velocity,
            max_angular_velocity=max_angular_velocity,
        )
        self._v0 = float(const_linear_velocity)
        self._waypoints = [tuple(wp) for wp in waypoints]
        self._goal_tol = float(goal_tolerance)
        self._target_idx = 0
        self._is_done = False
        
        # Potential Field parameters
        self._k_repel = float(repulsion_gain)
        self._D_safe = float(detection_radius)
        self._v_min = float(min_linear_velocity)
        self._debug_vectors = {} 

    def is_done(self) -> bool:
        return self._is_done

    def get_current_goal(self) -> Optional[Tuple[float, float]]:
        if self._is_done: return None
        return self._waypoints[self._target_idx]

    def _check_goal_arrival(self, x: float, y: float) -> None:
        if self._is_done: return
        gx, gy = self._waypoints[self._target_idx]
        dist_to_goal = np.hypot(gx - x, gy - y)
        if dist_to_goal <= self._goal_tol:
            print(f"[Navigator] Reached waypoint {self._target_idx}: ({gx:.2f}, {gy:.2f})")
            if self._target_idx < len(self._waypoints) - 1:
                self._target_idx += 1
            else:
                self._is_done = True
                print("[Navigator] Path complete.")

    def compute_control(
        self,
        state: Sequence[float],
        obstacle_state: Sequence[float],
        *_, **__
    ) -> Tuple[float, float]:
        """
        Calculates attractive force (goal) + repulsive force (obstacle).
        """
        x, y, theta = float(state[0]), float(state[1]), float(state[2])
        obs_x, obs_y = float(obstacle_state[0]), float(obstacle_state[1])

        self._check_goal_arrival(x, y)
        if self._is_done:
            return 0.0, 0.0

        # Goal (Attractive) Vector
        gx, gy = self._waypoints[self._target_idx]
        dx_goal, dy_goal = gx - x, gy - y
        
        # Obstacle (Repulsive) Vector
        dx_obs, dy_obs = obs_x - x, obs_y - y
        dist_obs = np.hypot(dx_obs, dy_obs)
        
        dx_repel, dy_repel = 0.0, 0.0
        v_final = self._v0 

        if 0 < dist_obs <= self._D_safe: 
            print(f"[Navigator] WARNING! Obstacle at {dist_obs:.2f}m")
            # Potential Field Repulsion Formula
            mag_repel = self._k_repel * (1.0 / dist_obs - 1.0 / self._D_safe)
            dx_repel = mag_repel * (-dx_obs / dist_obs)
            dy_repel = mag_repel * (-dy_obs / dist_obs)
            
            # Slow down near obstacle
            v_scale = (dist_obs / self._D_safe)
            v_final = max(self._v_min, self._v0 * v_scale)
            
        # Sum vectors
        dx_final, dy_final = dx_goal + dx_repel, dy_goal + dy_repel
        
        theta_star = float(np.arctan2(dy_final, dx_final))
        heading_err = _wrap_angle(theta_star - theta)
        w_final = self._k_h * heading_err
        
        self._debug_vectors = {
            "goal": (dx_goal, dy_goal),
            "repel": (dx_repel, dy_repel),
            "final": (dx_final, dy_final),
            "is_avoiding": (dist_obs <= self._D_safe and dist_obs > 0)
        }
        return self._clip(v_final, w_final)


# ==============================================================================
# --- EXTENSION 3: TRAJECTORY TRACKER (SPATIO-TEMPORAL) ---
# ==============================================================================

class TrajectoryTracker(UnicycleController):
    """
    Tracks a virtual reference point that moves along a path of waypoints 
    at a constant reference speed (v_ref).
    """

    def __init__(
        self,
        *,
        linear_velocity_gain: float = 1.5,   # k_v
        angular_velocity_gain: float = 3.0,  # k_h
        waypoints: Sequence[Tuple[float, float]] = [(1.0, 0.0)],
        reference_speed: float = 0.5,        # v_ref [m/s]
        max_linear_velocity: float = 0.8,
        max_angular_velocity: float = 2.5,
    ) -> None:
        super().__init__(
            linear_velocity_gain=linear_velocity_gain,
            angular_velocity_gain=angular_velocity_gain,
            max_linear_velocity=max_linear_velocity,
            max_angular_velocity=max_angular_velocity,
        )
        self._waypoints = [np.array(wp) for wp in waypoints]
        self._v_ref = float(reference_speed)
        self._is_done = False
        
        # Pre-calculate segment lengths and timing
        self._segment_distances = []
        self._segment_times = []
        self._cumulative_times = [0.0]
        
        total_time = 0.0
        for i in range(len(self._waypoints) - 1):
            p_start, p_end = self._waypoints[i], self._waypoints[i+1]
            dist = np.linalg.norm(p_end - p_start)
            time_for_segment = dist / self._v_ref if dist > 0 else 0
            
            self._segment_distances.append(dist)
            self._segment_times.append(time_for_segment)
            total_time += time_for_segment
            self._cumulative_times.append(total_time)
            
        self._total_path_time = total_time
        print(f"[TrajectoryTracker] Path initialized.")
        print(f"  Total distance: {sum(self._segment_distances):.2f}m")
        print(f"  Total time (at v_ref={self._v_ref}): {self._total_path_time:.2f}s")

    def is_done(self) -> bool:
        return self._is_done

    def _compute_reference_point(self, t: float) -> Tuple[float, float]:
        """Calculates the position (x_d, y_d) of the virtual reference point at time t."""
        if t >= self._total_path_time:
            self._is_done = True
            return self._waypoints[-1][0], self._waypoints[-1][1]
        
        # Determine which segment we are on
        segment_idx = -1
        for i in range(len(self._cumulative_times) - 1):
            if self._cumulative_times[i] <= t < self._cumulative_times[i+1]:
                segment_idx = i
                break
        
        if segment_idx == -1: segment_idx = 0

        p_start = self._waypoints[segment_idx]
        p_end = self._waypoints[segment_idx+1]
        time_on_segment = t - self._cumulative_times[segment_idx]
        total_time_for_segment = self._segment_times[segment_idx]
        
        # Interpolate
        alpha = 0.0
        if total_time_for_segment > 0:
            alpha = time_on_segment / total_time_for_segment

        p_ref = p_start + alpha * (p_end - p_start)
        return float(p_ref[0]), float(p_ref[1])

    def compute_control(self, state: Sequence[float], *, t: float) -> Tuple[float, float]:
        """Compute v, w to chase the moving reference."""
        goal_xy = self._compute_reference_point(t)
        if self._is_done:
            return 0.0, 0.0
        return self.go_to_point(state, goal_xy)
    
    def get_tracking_error(self, state: Sequence[float], t: float) -> float:
        """Helper to compute distance error from the virtual reference."""
        x, y = float(state[0]), float(state[1])
        ref_x, ref_y = self._compute_reference_point(t)
        if self._is_done: return 0.0
        return np.hypot(ref_x - x, ref_y - y)


# ==============================================================================
# --- EXTENSION 4: CORRIDOR FOLLOWER (Pure Reactive) ---
# ==============================================================================

class CorridorFollower(UnicycleController):
    """Navigates a corridor using 3 simulated 'raycast' sensors."""

    def __init__(
        self,
        *,
        linear_velocity_gain: float = 0.5,   # k_v
        angular_velocity_gain: float = 2.0,  # k_p
        min_linear_velocity: float = 0.1,
        max_linear_velocity: float = 0.8,
        max_angular_velocity: float = 2.5,
    ) -> None:
        super().__init__(
            linear_velocity_gain=0.0, 
            angular_velocity_gain=angular_velocity_gain,
            max_linear_velocity=max_linear_velocity,
            max_angular_velocity=max_angular_velocity,
        )
        self._k_v = float(linear_velocity_gain)
        self._v_min = float(min_linear_velocity)
        self.debug_rays = {'front': 0.0, 'front_left': 0.0, 'front_right': 0.0}

    def compute_control(
        self,
        state: Sequence[float],
        *,
        d_front: float,
        d_front_left: float,
        d_front_right: float
    ) -> Tuple[float, float]:
        """Calculates v and w based purely on sensor readings."""
        
        # 1. Velocity Control: Faster if front is clear
        v = self._k_v * d_front
        v = float(np.clip(v, self._v_min, self._v_max))
        
        # 2. Steering Control: Center between left and right walls
        steer_error = d_front_right - d_front_left
        omega = self._k_h * steer_error
        
        self.debug_rays = {
            'front': d_front,
            'front_left': d_front_left,
            'front_right': d_front_right
        }
        return self._clip(v, omega)


# ==============================================================================
# --- FINAL EXTENSION: HYBRID NAVIGATOR (Wall Follower Override) ---
# ==============================================================================
class HybridNavigator(TrajectoryTracker):
    """
    Hybrid Controller (Subsumption Architecture).
    
    Behaviors:
    1. PLANNER ("The Boss"): Follows the trajectory using TrajectoryTracker.
    2. SAFEGUARD ("The Bodyguard"): If an obstacle is too close, it overrides 
       the planner and switches to a reactive Wall Follower mode.

    Modes:
        - "catcher":
            The robot chases the moving reference (green target) and can
            boost its linear speed when it is far from it.
        - "perfect_line_follower":
            The robot ignores the target for planning and tries to stay
            as close as possible to the geometric line (path), with
            approximately constant linear speed.
    """

    def __init__(
        self,
        *,
        # --- Planner Parameters (Trajectory Tracker) ---
        linear_velocity_gain: float = 1.5,   # k_v (for v_goal)
        angular_velocity_gain: float = 3.0,  # k_h (for w_goal)
        waypoints: Sequence[Tuple[float, float]] = ((1.0, 0.0),),
        reference_speed: float = 0.5,        # v_ref
        
        # --- Safeguard Parameters (Reactive Wall Follower) ---
        reactive_angular_gain: float = 2.0,  # k_p_reactive
        safety_distance: float = 0.8,        # D_safe
        wall_follow_distance: float = 0.5,   # D_target (dist to keep from wall)
        reactive_speed: float = 0.3,         # v_reactive (constant panic speed)
        
        # --- Limits ---
        max_linear_velocity: float = 0.8,
        max_angular_velocity: float = 2.5,

        # --- Mode selection ---
        mode: str = "catcher",               # "catcher" or "perfect_line_follower"
    ) -> None:
        # Initialize the Planner (TrajectoryTracker)
        super().__init__(
            linear_velocity_gain=linear_velocity_gain,
            angular_velocity_gain=angular_velocity_gain,
            waypoints=waypoints,
            reference_speed=reference_speed,
            max_linear_velocity=max_linear_velocity,
            max_angular_velocity=max_angular_velocity,
        )
        
        # Initialize the Safeguard (Reactive Wall Follower)
        self._k_p_reactive = float(reactive_angular_gain)
        self._D_safe = float(safety_distance)
        self._D_target = float(wall_follow_distance)
        self._v_reactive = float(reactive_speed)

        # Mode of operation
        mode = str(mode).lower()
        if mode not in ("catcher", "perfect_line_follower"):
            mode = "catcher"
        self._mode = mode

    # --------------------------------------------------------------
    # Helper para el modo PERFECT LINE FOLLOWER
    # --------------------------------------------------------------
    def _closest_point_on_path(self, x: float, y: float) -> Tuple[float, float]:
        """
        Returns the closest point on the piecewise-linear path (list of waypoints)
        to the current position (x, y), by projecting onto each segment.
        Used in 'perfect_line_follower' mode to stay on the line.
        """
        p = np.array([x, y], dtype=float)
        best_p = self._waypoints[0]
        best_d2 = float("inf")

        for i in range(len(self._waypoints) - 1):
            a = self._waypoints[i]
            b = self._waypoints[i + 1]
            ab = b - a
            norm2 = float(np.dot(ab, ab))
            if norm2 <= 1e-9:
                continue  # ignore degenerate segment

            # Projection of p on segment [a, b]
            t_param = float(np.dot(p - a, ab) / norm2)
            t_param = float(np.clip(t_param, 0.0, 1.0))
            proj = a + t_param * ab

            d2 = float(np.sum((proj - p) ** 2))
            if d2 < best_d2:
                best_d2 = d2
                best_p = proj

        return float(best_p[0]), float(best_p[1])

    # --------------------------------------------------------------
    # Control principal (planner + safeguard)
    # --------------------------------------------------------------
    def compute_control(
        self,
        state: Sequence[float],
        *,
        t: float,
        d_front: float,
        d_front_left: float,
        d_front_right: float,
    ) -> Tuple[float, float]:
        """
        Compute (v, w) combining:
        - Planner (depending on mode: 'catcher' or 'perfect_line_follower')
        - Reactive safeguard (wall following) when an obstacle is too close.
        """
        # --- 1. PLANNER CALCULATIONS (Nominal Behavior) ---

        # El punto verde (rabbit) SIEMPRE se calcula para:
        #   - mover la zanahoria a lo largo de la trayectoria
        #   - saber cuándo termina (_is_done)
        ref_x, ref_y = self._compute_reference_point(t)
        if self._is_done:
            # Trayectoria terminada -> paramos el robot
            return 0.0, 0.0

        x, y, theta = float(state[0]), float(state[1]), float(state[2])

        if self._mode == "perfect_line_follower":
            # MODO 2: seguir la línea lo más perfectamente posible.
            # Ignoramos el target para el cálculo del punto de referencia:
            line_x, line_y = self._closest_point_on_path(x, y)

            # Usamos la misma ley que go_to_point, pero con v aproximadamente constante.
            v_tmp, w_line = self.go_to_point(state, (line_x, line_y))
            v_line = getattr(self, "_v_ref", abs(v_tmp))
            v_goal, w_goal = v_line, w_line

        else:
            # MODO 1 (por defecto): "catcher"
            # Perseguimos directamente el target que se mueve sobre la trayectoria.
            goal_xy = (ref_x, ref_y)
            v_goal, w_goal = self.go_to_point(state, goal_xy)

            # Distancia al target: si es grande, aumentamos v para "cazar" la zanahoria.
            dist_to_target = math.hypot(ref_x - x, ref_y - y)
            far_threshold = 0.5  # a partir de ~0.5 m comenzamos a dar un boost

            if dist_to_target > far_threshold:
                # Boost limitado (hasta 2x) y luego sat. por _clip()
                boost = min(2.0, 1.0 + (dist_to_target - far_threshold))
                v_goal = v_goal * boost

        # --- 2. SAFEGUARD ASSESSMENT ---
        d_min = min(d_front, d_front_left, d_front_right)

        # --- 3. ARBITRATION (Hard Switch) ---

        # CASE A: SAFE (d_min > D_safe)
        # El entorno está despejado -> seguimos al planner según el modo.
        if d_min > self._D_safe:
            return self._clip(v_goal, w_goal)

        # CASE B: DANGER (d_min <= D_safe)
        # Obstáculo cerca -> anulamos el planner y usamos modo reactivo.
        # ----------------------------------------------------------
        # REACTIVE LOGIC (Right-hand Wall Following)
        # ----------------------------------------------------------

        # 1. Steering (w): control proporcional sobre la distancia a la "pared"
        wall_error = self._D_target - d_front_right
        w_final = self._k_p_reactive * wall_error

        # 2. Velocity (v): velocidad de pánico...
        v_final = self._v_reactive

        # ... salvo si hay algo muy cerca delante.
        if d_front < self._D_safe:
            # Frenado proporcional para evitar colisión frontal
            v_scale = max(0.0, d_front / self._D_safe)  # escala 0..1
            v_final = self._v_reactive * v_scale

        # No permitimos marcha atrás en este esquema
        v_final = max(0.0, v_final)

        # Si por alguna razón acabamos con v = w = 0 (atrapado pegado a un obstáculo),
        # damos un pequeño giro en el sitio para "desatascar".
        if v_final == 0.0 and abs(w_final) < 1e-3 and d_min < self._D_safe:
            w_final = 0.5  # pequeño giro para buscar una salida

        return self._clip(v_final, w_final)
