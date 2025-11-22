# Robotics project: Line Following Unicycle Robot
## GitHub: robotics-line-following
### Group: Biddiscombe, Martinez, Spanò

Idea: simulate a unicycle robot that follows a line in a determined region.

Create a world bound, add a line (of type y = ax + b) randomly in the space, place the robot in random coordinates in the bounding box.

Simulation: robot goes towards line, follows it in bounds of world bound.


# From project proposal slides:
Project #4: line following with a unicycle
• Develop a control algorithm that enables a unicycle robot to follow a predefined
line on the ground
• Use the simulated unicycle robot seen during the classes
• Create a path (e.g. a black line on the floor); the robot starts off the line and must
reach the line and follow it until the end. The line acts as a reference.
• Compute a distance error 𝑒d (distance from the line) and a heading error 𝑒𝜃 (angle
difference with the line)
• Apply a constant speed 𝑣 = 𝑣0 , and a feedback control for the angular velocity
• Tune gains, compare results
• Plot analysis
• Possible extension: add trajectory tracking, add more lines, etc…


## Current idea:
Setup the environment to be an open space with a ranomly placed line and obstacles.

Create several iterations of the unicycle controller that follow the line according to different criteria. A target follows the line at a constant speed to indicate the "carrot" whilst the unicycle can be the "rabbit". 
 - Catcher: Unicycle controller that tries to stay on the target, avoiding obstacles only if necessary and accelerating to reach the carrot faster.
 - Perfect line follower: follows a path that covers as much as the line as possible, closely going around objects to return to the line and follow it closely, sacrificing time to catch the carrot.

We can evaluate the robot line followers according to various criteria, such as how long they manage to stay on the target, how much of the line they cover, if the target gets away, ...