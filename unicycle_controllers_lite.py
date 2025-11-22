from typing import Optional, Sequence, Tuple
import numpy as np
import math 

# ==============================================================================
# --- BASE CLASS: KINEMATICS & BASIC CONTROL ---
# ==============================================================================

def _wrap_angle(angle: float) -> float:
    """
    Wraps an angle to the [-pi, pi] interval.
    Essential for calculating the smallest difference between two headings.
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

class UnicycleController:
    """
    Base class containing shared logic for unicycle kinematic models.
    Implements velocity limits and a proportional 'go-to-point' control law.
    """

    def __init__(
        self,
        *,
        linear_velocity_gain: float = 0.5,   # k_v: Proportional gain for linear speed
        angular_velocity_gain: float = 1.0,  # k_h: Proportional gain for heading
        max_linear_velocity: float = 0.5,    # Saturation limit for v
        max_angular_velocity: float = 1.5,   # Saturation limit for omega
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
        """Updates saturation limits dynamically."""
        self._v_max = float(linear)
        self._w_max = float(angular)

    def set_gains(self, k_v: float, k_h: float) -> None:
        """Updates control gains dynamically."""
        self._k_v = float(k_v)
        self._k_h = float(k_h)

    def _clip(self, v: float, omega: float) -> Tuple[float, float]:
        """
        Saturates the calculated control outputs to the physical limits of the robot.
        Returns: (clamped_v, clamped_omega)
        """
        v_c = float(np.clip(v, -self._v_max, self._v_max))
        w_c = float(np.clip(omega, -self._w_max, self._w_max))
        return v_c, w_c

    def go_to_point(self, current_state: Sequence[float], goal_xy: Tuple[float, float]) -> Tuple[float, float]:
        """
        Calculates control inputs to drive the unicycle toward a static 2D goal 
        using a simple proportional control law.

        Args:
            current_state: [x, y, theta] of the robot.
            goal_xy: [x_goal, y_goal] of the target.

        Returns:
            (v, omega): Linear and angular velocities.
        """
        x, y, theta = float(current_state[0]), float(current_state[1]), float(current_state[2])
        gx, gy = float(goal_xy[0]), float(goal_xy[1])

        # Calculate error vector
        dx = gx - x
        dy = gy - y
        
        # rho: Euclidean distance to the goal (Position Error)
        rho = float(np.hypot(dx, dy)) 
        
        # theta_star: The angle of the vector pointing to the goal
        theta_star = float(np.arctan2(dy, dx)) 
        
        # heading_err: Difference between desired heading and current heading
        heading_err = _wrap_angle(theta_star - theta)

        # Proportional Control Law
        v = self._k_v * rho             # Drive faster if goal is far
        omega = self._k_h * heading_err # Turn faster if heading error is large

        return self._clip(v, omega)

    def compute_control(self, state: Sequence[float], *args, **kwargs) -> Tuple[float, float]:  # pragma: no cover
        """Interface method to be implemented by subclasses."""
        raise NotImplementedError


# ==============================================================================
# --- EXTENSION 1: TRAJECTORY TRACKER (SPATIO-TEMPORAL) ---
# ==============================================================================

class TrajectoryTracker(UnicycleController):
    """
    Tracks a virtual reference point that moves along a path of waypoints 
    at a constant reference speed (v_ref).
    """

    def __init__(
        self,
        *,
        linear_velocity_gain: float = 1.5,   # Higher gain for tracking
        angular_velocity_gain: float = 3.0,  # Higher gain for sharp turns
        waypoints: Sequence[Tuple[float, float]] = [(1.0, 0.0)],
        reference_speed: float = 0.5,        # v_ref [m/s] of the virtual rabbit
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
            
            # Time required to traverse this segment at v_ref
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
        """Returns True if the virtual reference has reached the end of the path."""
        return self._is_done

    def _compute_reference_point(self, t: float) -> Tuple[float, float]:
        """
        Calculates the position (x_d, y_d) of the virtual reference point at time t.
        It interpolates between waypoints based on elapsed time.
        """
        # If time exceeds total path time, stay at the last waypoint
        if t >= self._total_path_time:
            self._is_done = True
            return self._waypoints[-1][0], self._waypoints[-1][1]
        
        # Find which segment corresponds to the current time t
        segment_idx = -1
        for i in range(len(self._cumulative_times) - 1):
            if self._cumulative_times[i] <= t < self._cumulative_times[i+1]:
                segment_idx = i
                break
        
        if segment_idx == -1: segment_idx = 0

        # Linear Interpolation (Lerp) along the active segment
        p_start = self._waypoints[segment_idx]
        p_end = self._waypoints[segment_idx+1]
        
        time_on_segment = t - self._cumulative_times[segment_idx]
        total_time_for_segment = self._segment_times[segment_idx]
        
        alpha = 0.0
        if total_time_for_segment > 0:
            alpha = time_on_segment / total_time_for_segment # 0.0 to 1.0

        p_ref = p_start + alpha * (p_end - p_start)
        return float(p_ref[0]), float(p_ref[1])

    def compute_control(self, state: Sequence[float], *, t: float) -> Tuple[float, float]:
        """
        Calculates (v, w) to chase the moving reference point.
        """
        goal_xy = self._compute_reference_point(t)
        if self._is_done:
            return 0.0, 0.0
        # Use the base class Proportional controller to chase the moving point
        return self.go_to_point(state, goal_xy)
    
    def get_tracking_error(self, state: Sequence[float], t: float) -> float:
        """Helper to compute Euclidean distance between robot and reference point."""
        x, y = float(state[0]), float(state[1])
        ref_x, ref_y = self._compute_reference_point(t)
        if self._is_done: return 0.0
        return np.hypot(ref_x - x, ref_y - y)


# ==============================================================================
# --- EXTENSION 2: HYBRID NAVIGATOR (WALL FOLLOWER OVERRIDE) ---
# ==============================================================================

class HybridNavigator(TrajectoryTracker):
    """
    Hybrid Controller (Subsumption Architecture).
    
    Behaviors:
    1. PLANNER ("The Boss"): Follows the trajectory using TrajectoryTracker.
    2. SAFEGUARD ("The Bodyguard"): If an obstacle is too close, it overrides 
       the planner and switches to a reactive Wall Follower mode.
    """

    def __init__(
        self,
        *,
        # --- Planner Parameters (Trajectory Tracker) ---
        linear_velocity_gain: float = 1.5,   # k_v (for v_goal)
        angular_velocity_gain: float = 3.0,  # k_h (for w_goal)
        waypoints: Sequence[Tuple[float, float]] = [(1.0, 0.0)],
        reference_speed: float = 0.5,        # v_ref
        
        # --- Safeguard Parameters (Reactive Wall Follower) ---
        reactive_angular_gain: float = 2.0,  # k_p_reactive (Corrective steering)
        safety_distance: float = 0.8,        # D_safe: Threshold to trigger panic mode
        wall_follow_distance: float = 0.5,   # D_target: Desired distance from wall
        reactive_speed: float = 0.3,         # v_reactive: Constant speed during panic
        
        # --- Physical Limits ---
        max_linear_velocity: float = 0.8,
        max_angular_velocity: float = 2.5,
    ) -> None:
        # Initialize the Planner
        super().__init__(
            linear_velocity_gain=linear_velocity_gain, angular_velocity_gain=angular_velocity_gain,
            waypoints=waypoints, reference_speed=reference_speed,
            max_linear_velocity=max_linear_velocity, max_angular_velocity=max_angular_velocity)
        
        # Initialize the Safeguard
        self._k_p_reactive = float(reactive_angular_gain)
        self._D_safe = float(safety_distance)
        self._D_target = float(wall_follow_distance)
        self._v_reactive = float(reactive_speed)

    def compute_control(
        self,
        state: Sequence[float],
        *,
        t: float,
        d_front: float,
        d_front_left: float,
        d_front_right: float
    ) -> Tuple[float, float]:
        """
        Computes control output based on sensor data.
        Arbitrates between Path Following and Reactive Obstacle Avoidance.
        """
        
        # --- 1. PLANNER CALCULATIONS (Nominal Behavior) ---
        goal_xy = self._compute_reference_point(t)
        if self._is_done: return 0.0, 0.0
        
        # Calculate what the planner WANTS to do
        (v_goal, w_goal) = self.go_to_point(state, goal_xy)
        
        # --- 2. SAFEGUARD ASSESSMENT ---
        # Check the minimum distance reported by front sensors
        d_min = min(d_front, d_front_left, d_front_right)
        
        # --- 3. ARBITRATION (Hard Switch) ---
        
        # CASE A: SAFE (d_min > D_safe)
        # The environment is clear. Obey the "Boss" (Planner) completely.
        if d_min > self._D_safe:
            return self._clip(v_goal, w_goal)
        
        # CASE B: DANGER (d_min <= D_safe)
        # Obstacle detected! IGNORE the Planner.
        # Engage "Bodyguard" mode (Reactive Wall Following).
        
        # --- REACTIVE LOGIC (Right-hand Wall Following) ---
        # Goal: Keep the wall on the RIGHT at distance D_target.
        
        # A. Steering Control (Angular Velocity):
        # Error = Desired_Distance - Actual_Distance_Right
        # If wall is too far (Error > 0) -> Turn Right (negative w? depends on frame)
        # NOTE: Assuming standard unicycle, positive w turns Left.
        # If wall is too close (d_right < target), error is positive.
        # The logic below assumes:
        #   If d_right is SMALL, we need to turn LEFT (positive w) to move away.
        #   Error = D_target - d_front_right.
        #   If d_front_right is 0.3 (too close) vs target 0.5 -> Error = 0.2 -> w > 0 (Turn Left). Correct.
        wall_error = self._D_target - d_front_right
        w_final = self._k_p_reactive * wall_error
        
        # B. Speed Control (Linear Velocity):
        # Default to constant reactive speed...
        v_final = self._v_reactive
        
        # ... but brake if there is a wall directly IN FRONT.
        if d_front < self._D_safe:
             # Proportional braking: scale speed from 0 to 1 based on remaining distance
             v_scale = max(0.0, d_front / self._D_safe) 
             v_final = self._v_reactive * v_scale
             
        v_final = max(0.0, v_final) # Ensure we don't reverse
        
        return self._clip(v_final, w_final)