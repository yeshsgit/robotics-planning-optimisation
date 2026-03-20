''''
Purpose of the code is to illustrate the A* algorithm with 2D 2 link simulation. This will show 2 maps, one in x and y coordinate
and one in the 2 angles for each of the links.

angle 1, or theta 1 will be the link closest to the origin/stationary point of the robot and angle 2, etc will be the one further away
both have a set length of 10 units

angles are in radians because that's what numpy uses

first part of the code makes fk, ik functions
then, it makes a grid in the 

'''

import matplotlib.pyplot as plt
import numpy as np
from collision_api import collision

l1=10
l2=10

angle1Division = 50
angle2Division = 50

#the following function gives the same thing
def fk_linear(angle):
    '''
    input: angle (vector of angle1, angle2)
    output: position (x, y)
    '''
    x = l1 * np.cos(angle[0]) + l2 * np.cos(angle[0] + angle[1])
    y = l1 * np.sin(angle[0]) + l2 * np.sin(angle[0] + angle[1])
    cartesian = [x, y]
    return cartesian

def fk(angle):
    '''
    input: angle (vector of angle1, angle2)
    output: position (x, y)

    however, the 0, 0 to 1, 1 of T gives homogenous transformation matrix
    '''
    theta1 = angle[0]
    theta2 = angle[1]
    T1 = np.array([
        [np.cos(theta1), -np.sin(theta1), l1*np.cos(theta1)],
        [np.sin(theta1),  np.cos(theta1), l1*np.sin(theta1)],
        [0, 0, 1]
    ])
    
    T2 = np.array([
        [np.cos(theta2), -np.sin(theta2), l2*np.cos(theta2)],
        [np.sin(theta2),  np.cos(theta2), l2*np.sin(theta2)],
        [0, 0, 1]
    ])

    T = T1 @ T2
    x = T[0, 2]
    y = T[1, 2]
    return [x, y]

def ik(coords):
    '''
    input: the position vector
    output: [angle1, angle2], [angle1, angle2] for both orientations

    
    '''
    x = coords[0]
    y = coords[1]
    # Distance squared
    r2 = x**2 + y**2
    
    if np.sqrt(r2) > l1+l2:
        print("outside the solution")
        return [0, 0], [0, 0]

    # Compute cos(theta2)
    cos_theta2 = (r2 - l1**2 - l2**2) / (2 * l1 * l2)
    
    # Clamp for numerical safety
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
    
    # Compute sin(theta2) magnitude
    sin_theta2_pos = np.sqrt(1 - cos_theta2**2)
    sin_theta2_neg = -sin_theta2_pos
    
    # Two possible theta2 solutions
    theta2_up = np.arctan2(sin_theta2_pos, cos_theta2)
    theta2_down = np.arctan2(sin_theta2_neg, cos_theta2)
    
    # Corresponding theta1 solutions
    theta1_up = np.arctan2(y, x) - np.arctan2(
        l2 * sin_theta2_pos,
        l1 + l2 * cos_theta2
    )
    
    theta1_down = np.arctan2(y, x) - np.arctan2(
        l2 * sin_theta2_neg,
        l1 + l2 * cos_theta2
    )
    
    return [theta1_up, theta2_up], [theta1_down, theta2_down]

print(fk([np.pi, 0]))
print(fk_linear([np.pi, 0]))
ans1, ans2 = ik([-20, 0])
print(ans1)
print(ans2)


###here is setting up the space and making functions for the A* algorithm.
###it requires a neighbor function, cost function (this will be 1 if they are neighbors)

space1 = np.linspace(0, 2*np.pi, num=angle1Division, endpoint=False)
space2 = np.linspace(0, 2*np.pi, num=angle2Division, endpoint=False)

workspace = [(x, y) for x in space1 for y in space2]

def cost():
    return 1

def neighbor(pos1, pos2):
    '''
    input: workspace coords
    output: boolean

    returns true if they are neighbors. it will be in 8 directions and not 4, because in the same amount of time,
    the robot should be able to move to 8 different points
    '''
    # Find positions using the 1D arrays space1 and space2
    #This works for some reason
    i1 = np.argmin(np.abs(space1 - pos1[0]))  # index in space1
    j1 = np.argmin(np.abs(space2 - pos1[1]))  # index in space2
    i2 = np.argmin(np.abs(space1 - pos2[0]))  # index in space1
    j2 = np.argmin(np.abs(space2 - pos2[1]))  # index in space2
    
    # Handle x dimension (space1) - 0 and 2π are the same point
    if (i1 == 0 and i2 == angle1Division - 1) or (i1 == angle1Division - 1 and i2 == 0):
        dx = 0  # They are the same point (0 and 2π are equivalent)
    elif (i1 == 0 and i2 == angle1Division - 2) or (i1 == angle1Division - 1 and i2 == 1):
        dx = 1  # Neighbors with wrapping (0 adjacent to one step before 2π)
    elif (i1 == angle1Division - 2 and i2 == 0) or (i1 == 1 and i2 == angle1Division - 1):
        dx = 1  # Neighbors with wrapping (one step before 2π adjacent to 0)
    else:
        dx = abs(i1 - i2)
    
    # Handle y dimension (space2) - 0 and 2π are the same point
    if (j1 == 0 and j2 == angle2Division - 1) or (j1 == angle2Division - 1 and j2 == 0):
        dy = 0  # They are the same point (0 and 2π are equivalent)
    elif (j1 == 0 and j2 == angle2Division - 2) or (j1 == angle2Division - 1 and j2 == 1):
        dy = 1  # Neighbors with wrapping (0 adjacent to one step before 2π)
    elif (j1 == angle2Division - 2 and j2 == 0) or (j1 == 1 and j2 == angle2Division - 1):
        dy = 1  # Neighbors with wrapping (one step before 2π adjacent to 0)
    else:
        dy = abs(j1 - j2)
    
    # Return True if they are neighbors (at most 1 away in both dimensions, but not both 0)
    return (dx <= 1 and dy <= 1) and (dx != 0 or dy != 0)

def heuristic(pos, goal):
    '''
    input: position, goal
    output: distance huristic. now this is the number of minimum steps using the neighbor function
    so 8 directions

    This works by:
    Finds the grid indices of both positions in the 2D workspace
    Handles the case where 0 and 2π are the same point (distance = 0)
    Compares direct and wrapped paths for each dimension:
    Direct path: straight distance
    Wrapped path: going through the wrap (accounting that 0 and the last index are the same)
    Takes the minimum distance for each dimension
    Returns max(dx, dy) for 8-directional movement (Chebyshev distance)
    '''

    
    # Find positions using the 1D arrays space1 and space2
    i1 = np.argmin(np.abs(space1 - pos[0]))  # index in space1
    j1 = np.argmin(np.abs(space2 - pos[1]))  # index in space2
    i2 = np.argmin(np.abs(space1 - goal[0]))  # index in space1
    j2 = np.argmin(np.abs(space2 - goal[1]))  # index in space2
    
    # Handle x dimension - check if 0 and 2π are the same point
    if (i1 == 0 and i2 == angle1Division - 1) or (i1 == angle1Division - 1 and i2 == 0):
        dx = 0  # Same point
    else:
        dx_direct = abs(i1 - i2)
        # For wrapping: consider going the other way around
        # If i1=0 and i2=48, can go 0->49(same as 0)->48 = 1 step, or 0->1->...->48 = 48 steps
        # So wrapped distance = min(dx_direct, angle1Division - dx_direct - 1 + 1) = min(dx_direct, angle1Division - dx_direct)
        # But need to account for the fact that 0 and last index are same
        dx_wrap = angle1Division - dx_direct - 1  # Going through the wrap (excluding the same point)
        dx = min(dx_direct, dx_wrap)
    
    # Handle y dimension - check if 0 and 2π are the same point
    if (j1 == 0 and j2 == angle2Division - 1) or (j1 == angle2Division - 1 and j2 == 0):
        dy = 0  # Same point
    else:
        dy_direct = abs(j1 - j2)
        # For wrapping: consider going the other way around
        dy_wrap = angle2Division - dy_direct - 1  # Going through the wrap (excluding the same point)
        dy = min(dy_direct, dy_wrap)
    
    # For 8-directional movement, minimum steps = max(|dx|, |dy|)
    return max(dx, dy) + min(dx, dy) * 0.25
'''
def astar(start, goal, initial_angles=None, final_angles=None):
    
    A* (A-star) pathfinding algorithm
    
    Procedure:
    1. Convert start and goal positions from (x, y) coordinates to workspace indices
       - If initial_angles and final_angles are provided, use them directly
       - Otherwise, use ik() to convert (x, y) -> [angle1, angle2] and try all combinations
       - Find closest workspace indices using argmin on space1 and space2 arrays
    2. Initialize open set with start position, closed set empty, g_score and f_score dictionaries
    3. While open set is not empty:
       a. Get node with lowest f_score from open set
       b. Move node from open to closed set
       c. If node is goal, reconstruct and return path
       d. For each neighbor of current node:
          - Calculate tentative g_score (g_score[current] + cost())
          - If neighbor not in closed set and (not in open set or tentative g_score < g_score[neighbor]):
             - Add/update neighbor in open set with tentative g_score
             - Set f_score[neighbor] = g_score[neighbor] + huristic(neighbor, goal)
             - Set parent[neighbor] = current node
    4. If goal not reached, return None (no path found)
    
    Input:
       start: [x, y] coordinates of start position
       goal: [x, y] coordinates of goal position
       initial_angles: optional [angle1, angle2] for start position (if None, uses ik() to find solutions)
       final_angles: optional [angle1, angle2] for goal position (if None, uses ik() to find solutions)
    
    Output:
       path: list of workspace coordinates [(x1, y1), (x2, y2), ...] representing the motion series
       or None if no path exists
    
    # Step 1: Convert start and goal positions from (x, y) coordinates to workspace indices
    if initial_angles is not None and final_angles is not None:
        # Use provided angles directly
        start_angles_list = [initial_angles]
        goal_angles_list = [final_angles]
    else:
        # Convert (x, y) to angles using ik() - ik returns two solutions (elbow up/down)
        start_angles_sol1, start_angles_sol2 = ik(start)
        goal_angles_sol1, goal_angles_sol2 = ik(goal)
        start_angles_list = [start_angles_sol1, start_angles_sol2]
        goal_angles_list = [goal_angles_sol1, goal_angles_sol2]
    
    # Try all combinations of solutions and find the shortest path
    # This handles cases where one solution might be closer in workspace than the other
    best_path = None
    best_path_length = float('inf')
    
    # Try all combinations: if angles provided, only one combination; otherwise all 4
    for start_angles in start_angles_list:
        for goal_angles in goal_angles_list:
            # Find closest workspace indices for start angles
            i_start = np.argmin(np.abs(space1 - start_angles[0]))
            j_start = np.argmin(np.abs(space2 - start_angles[1]))
            start_idx = i_start * angle2Division + j_start
            
            # Find closest workspace indices for goal angles
            i_goal = np.argmin(np.abs(space1 - goal_angles[0]))
            j_goal = np.argmin(np.abs(space2 - goal_angles[1]))
            goal_idx = i_goal * angle2Division + j_goal
            
            # Run A* for this combination
            path = astar_helper(start_idx, goal_idx)
            
            # Keep track of shortest path
            if path is not None and len(path) < best_path_length:
                best_path = path
                best_path_length = len(path)
    
    return best_path

def astar_helper(start_idx, goal_idx):
    
    Helper function that performs A* search given workspace indices
    This is called for each combination of IK solutions
    
    
    # Step 2: Initialize data structures
    open_set = {start_idx}  # Set of nodes to explore
    closed_set = set()  # Set of nodes already explored
    g_score = {start_idx: 0}  # Actual cost from start to each node
    f_score = {start_idx: huristic(workspace[start_idx], workspace[goal_idx])}  # Estimated total cost (g + h)
    parent = {start_idx: None}  # Parent node for path reconstruction (start has no parent)
    
    # Step 3: A* main loop
    while open_set:
        # 3a: Get node with lowest f_score from open set
        current_idx = min(open_set, key=lambda idx: f_score[idx])
        
        # 3b: Move node from open to closed set
        open_set.remove(current_idx)
        closed_set.add(current_idx)
        
        # 3c: If node is goal, reconstruct and return path
        if current_idx == goal_idx:
            # Reconstruct path by following parent pointers from goal back to start
            path = []
            node = goal_idx
            while node is not None:
                path.append(workspace[node])
                if node == start_idx:
                    break  # Reached start, path is complete
                node = parent.get(node, None)
            # Reverse to get path from start to goal
            path.reverse()
            return path
        
        # 3d: For each neighbor of current node
        current_pos = workspace[current_idx]
        i_current = current_idx // angle2Division
        j_current = current_idx % angle2Division
        
        # Check all 8 neighbors (including wrapping)
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue  # Skip self
                
                # Calculate neighbor indices with wrapping
                i_neighbor = (i_current + di) % angle1Division
                j_neighbor = (j_current + dj) % angle2Division
                neighbor_idx = i_neighbor * angle2Division + j_neighbor
                neighbor_pos = workspace[neighbor_idx]
                
                # Check if neighbor
                if not neighbor(workspace[current_idx], neighbor_pos):
                    continue
                
                # Calculate tentative g_score
                tentative_g = g_score[current_idx] + cost()
                
                # If neighbor not explored and (not in open or better path found)
                if neighbor_idx not in closed_set and (neighbor_idx not in open_set or tentative_g < g_score.get(neighbor_idx, float('inf'))):
                    # Add/update neighbor in open set
                    open_set.add(neighbor_idx)
                    g_score[neighbor_idx] = tentative_g
                    f_score[neighbor_idx] = tentative_g + huristic(neighbor_pos, workspace[goal_idx])
                    parent[neighbor_idx] = current_idx
    
    # Step 4: If goal not reached, return None
    return None

print()
print(astar([-9.592057023943893, -5.239353470528048], [4.479655557807313, -5.936719958160177]))'''
import heapq
import numpy as np


def astar(start, goal, initial_angles=None, final_angles=None):

    def normalize(theta):
        return theta % (2 * np.pi)

    # ----------------------------
    # Resolve start configuration
    # ----------------------------
    if initial_angles is not None:
        theta1_s, theta2_s = initial_angles
    else:
        sol1, sol2 = ik(start)
        theta1_s, theta2_s = sol1

    theta1_s = normalize(theta1_s)
    theta2_s = normalize(theta2_s)

    # ----------------------------
    # Resolve goal configuration
    # ----------------------------
    if final_angles is not None:
        theta1_g, theta2_g = final_angles
    else:
        sol1, sol2 = ik(goal)
        theta1_g, theta2_g = sol1

    theta1_g = normalize(theta1_g)
    theta2_g = normalize(theta2_g)

    # ----------------------------
    # Convert angle → index
    # ----------------------------
    def angle_to_index(theta1, theta2):
        i = np.argmin(np.abs(space1 - theta1))
        j = np.argmin(np.abs(space2 - theta2))
        return (i, j)

    start_idx = angle_to_index(theta1_s, theta2_s)
    goal_idx = angle_to_index(theta1_g, theta2_g)

    # ----------------------------
    # Heuristic wrapper
    # ----------------------------
    def h(idx):

        theta1 = space1[idx[0]]
        theta2 = space2[idx[1]]

        goal_theta1 = space1[goal_idx[0]]
        goal_theta2 = space2[goal_idx[1]]

        return heuristic((theta1, theta2),
                         (goal_theta1, goal_theta2))

    # ----------------------------
    # A* Search
    # ----------------------------
    open_heap = []
    heapq.heappush(open_heap, (h(start_idx), start_idx))

    came_from = {}
    g_score = {start_idx: 0}

    DIRECTIONS = [
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, 1), (1, -1),
        (-1, 1), (-1, -1),
    ]

    while open_heap:

        current_f, current = heapq.heappop(open_heap)

        # Skip stale entries (critical for optimality)
        if current_f > g_score[current] + h(current):
            continue

        if current == goal_idx:
            return reconstruct_path(came_from, current)

        for di, dj in DIRECTIONS:

            ni = (current[0] + di) % angle1Division
            nj = (current[1] + dj) % angle2Division

            neighbor = (ni, nj)

            theta1 = space1[ni]
            theta2 = space2[nj]

            if collision(theta1, l1, theta2, l2):
                #print("collision detected")
                continue
            else:
                tentative_g = g_score[current] + cost()

                if neighbor not in g_score or tentative_g < g_score[neighbor]:

                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g

                    f = tentative_g + h(neighbor)
                    heapq.heappush(open_heap, (f, neighbor))

    return None


def reconstruct_path(came_from, current):

    path = []

    while current in came_from:
        path.append((space1[current[0]],
                     space2[current[1]]))
        current = came_from[current]

    path.append((space1[current[0]],
                 space2[current[1]]))

    path.reverse()
    return path

print(astar([10, 10], [7, 6]))