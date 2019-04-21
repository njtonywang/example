# usage:  python visualize_map.py obstacles_file start_goal_file

from __future__ import division
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
import random, math
from scipy.spatial import ConvexHull
import time
from shapely.geometry import Point, LineString, Polygon


def build_obstacle_course(obstacle_path, ax):
    vertices = list()
    codes = [Path.MOVETO]
    with open(obstacle_path) as f:
        quantity = int(f.readline())
        lines = 0
        for line in f:
            coordinates = tuple(map(int, line.strip().split(' ')))
            if len(coordinates) == 1:
                codes += [Path.MOVETO] + [Path.LINETO]*(coordinates[0]-1) + [Path.CLOSEPOLY]
                vertices.append((0,0)) #Always ignored by closepoly command
            else:
                vertices.append(coordinates)
    vertices.append((0,0))
    vertices = np.array(vertices, float)
    path = Path(vertices, codes)
    pathpatch = patches.PathPatch(path, facecolor='None', edgecolor='xkcd:violet')

    ax.add_patch(pathpatch)
    ax.set_title('Sample-Based Motion Planning')

    ax.dataLim.update_from_data_xy(vertices)
    ax.autoscale_view()
    ax.invert_yaxis()

    return path

def add_start_and_goal(start_goal_path, ax):
    start, goal = None, None
    with open(start_goal_path) as f:
        start = tuple(map(int, f.readline().strip().split(' ')))
        goal  = tuple(map(int, f.readline().strip().split(' ')))

    ax.add_patch(patches.Circle(start, facecolor='xkcd:bright green'))
    ax.add_patch(patches.Circle(goal, facecolor='xkcd:fuchsia'))

    return start, goal

def read_obstacles(path):
    lines = []
    with open(path, "r") as file:
        for line in file:
            lines.append(line.strip())
    
    output = []
    total_obs = int(lines[0])
    line = 1
    for i in range(total_obs):
        vertices_count = int(lines[line])
        line += 1
        vertices = []
        for j in range(vertices_count):
            pair = lines[line].split()
            pair[0] = float(pair[0])
            pair[1] = float(pair[1])
            vertices.append(pair)  # each (x,y) vertices pair
            line += 1
        output.append(vertices)
    return output

def rrt(start, goal, obstacles, plt, step_size=20, bias=0.05):
    '''
    Input:
        start: tuple(x, y)
        goal: tuple(x, y)
        obstacles: a list of obstacle, each obstacle is a list of points [x,y]
    Output:
        number of nodes in the tree: int
    '''
    plt.ion()
    polygons = [Polygon(obs) for obs in obstacles]

    V = set() # a set of tuple p = (x, y)
    parent = {} # record parent node to draw path
    V.add(start)
    goal_p = Point(goal[0], goal[1])
    while True:
        # sample random
        p_rand = goal
        if random.random() > bias:
            p_rand = (random.random() * 600, random.random() * 600)

        # find nearest
        dist = float('inf')
        nearest_v = None
        for v in V:
            tmp_dist = (v[0]-p_rand[0])**2 + (v[1]-p_rand[1])**2
            if tmp_dist < dist:
                dist = tmp_dist
                nearest_v = v
        dist = math.sqrt(dist)
        if dist < step_size:
            continue

        # generate new point
        new_x = nearest_v[0] + step_size * (p_rand[0]-nearest_v[0]) / dist
        new_y = nearest_v[1] + step_size * (p_rand[1]-nearest_v[1]) / dist
        new_v = (new_x, new_y)
        new_p = Point(new_x, new_y)

        # judge intersect
        nearest_p = Point(nearest_v[0], nearest_v[1])
        new_line = LineString([nearest_p, new_p])
        if is_intersect(new_line, polygons):
            continue
        V.add(new_v)
        parent[new_v] = nearest_v
        plt.plot([nearest_v[0], new_x], [nearest_v[1], new_y], marker='.', color='b')
        plt.pause(0.01)

        # judge goal state
        dist_goal = math.sqrt((new_v[0]-goal[0])**2 + (new_v[1]-goal[1])**2)
        goal_line = LineString([goal_p, new_p])
        if not is_intersect(goal_line, polygons) and dist_goal < step_size:
            plt.plot([goal[0], new_x], [goal[1], new_y], marker='.', color='b')
            parent[goal] = new_v
            break

    # plot path
    cur = goal
    while True:
        p = parent[cur]
        plt.plot([cur[0], p[0]], [cur[1], p[1]], marker='.', color='r')
        if p == start:
            break
        cur = p
    
    plt.ioff()
    return len(V)

def is_intersect(line, polygons):
    '''
    Return whether the line intersects with any obstacle
    '''
    for polygon in polygons:
        if line.intersects(polygon):
            return True
    return False

def bi_rrt(start, goal, obstacles, plt, step_size=20, bias=0.05):
    '''
    Input:
        start: tuple(x, y)
        goal: tuple(x, y)
        obstacles: a list of obstacle, each obstacle is a list of points [x,y]
    Output:
        number of nodes in the tree: int
    '''
    plt.ion()
    polygons = [Polygon(obs) for obs in obstacles]

    V_0 = set() # tree a: a set of tuple p = (x, y)
    V_1 = set() # tree b: a set of tuple p = (x, y)
    parent = {} # record parent node to draw path
    V_0.add(start)
    V_1.add(goal)
    V = [V_0, V_1]
    colors = ['b', 'g']
    cur_tree = 0
    path_node_1, path_node_2 = None, None
    while True:
        # sample random
        p_rand = goal
        if cur_tree == 1:
            p_rand = start
        if random.random() > bias:
            p_rand = (random.random() * 600, random.random() * 600)
    
        # find nearest
        dist = float('inf')
        nearest_v = None
        for v in V[cur_tree]:
            tmp_dist = (v[0]-p_rand[0])**2 + (v[1]-p_rand[1])**2
            if tmp_dist < dist:
                dist = tmp_dist
                nearest_v = v
        dist = math.sqrt(dist)
        if dist < step_size:
            continue

        # generate new point
        new_x = nearest_v[0] + step_size * (p_rand[0]-nearest_v[0]) / dist
        new_y = nearest_v[1] + step_size * (p_rand[1]-nearest_v[1]) / dist
        new_v = (new_x, new_y)
        new_p = Point(new_x, new_y)

        # judge intersect
        nearest_p = Point(nearest_v[0], nearest_v[1])
        new_line = LineString([nearest_p, new_p])
        if is_intersect(new_line, polygons):
            continue
        V[cur_tree].add(new_v)
        parent[new_v] = nearest_v
        plt.plot([nearest_v[0], new_x], [nearest_v[1], new_y], marker='.', color=colors[cur_tree])
        plt.pause(0.01)

        # find nearest in the other tree
        dist = float('inf')
        nearest_v = None
        for v in V[1-cur_tree]:
            tmp_dist = (v[0]-new_x)**2 + (v[1]-new_y)**2
            if tmp_dist < dist:
                dist = tmp_dist
                nearest_v = v
        dist = math.sqrt(dist)

        # connect two trees
        nearest_p = Point(nearest_v[0], nearest_v[1])
        new_line = LineString([nearest_p, new_p])
        if is_intersect(new_line, polygons): # extend towards the new point
            while True:
                # generate new point
                extend_x = nearest_v[0] + step_size * (new_x-nearest_v[0]) / dist
                extend_y = nearest_v[1] + step_size * (new_y-nearest_v[1]) / dist
                extend_v = (extend_x, extend_y)
                extend_p = Point(extend_x, extend_y)

                # judge intersect
                extend_line = LineString([nearest_p, extend_p])
                if is_intersect(extend_line, polygons):
                    break
                V[1-cur_tree].add(extend_v)
                parent[extend_v] = nearest_v
                plt.plot([nearest_v[0], extend_x], [nearest_v[1], extend_y], marker='.', color=colors[1-cur_tree])
                plt.pause(0.01)
                # update iteratively
                nearest_v = extend_v
                dist = math.sqrt((extend_v[0]-new_x)**2 + (extend_v[1]-new_y)**2)
        else: # find the final solution
            plt.plot([nearest_v[0], new_x], [nearest_v[1], new_y], marker='.', color='r')
            path_node_1, path_node_2 = nearest_v, new_v
            break
        
        cur_tree = 1 - cur_tree # exchange tree

    # plot path
    cur = path_node_1
    while True:
        p = parent[cur]
        plt.plot([cur[0], p[0]], [cur[1], p[1]], marker='.', color='r')
        if p == start or p == goal:
            break
        cur = p

    cur = path_node_2
    while True:
        p = parent[cur]
        plt.plot([cur[0], p[0]], [cur[1], p[1]], marker='.', color='r')
        if p == start or p == goal:
            break
        cur = p
    
    plt.ioff()
    return len(V[0]) + len(V[1])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('obstacle_path',
                        help="File path for obstacle set")
    parser.add_argument('start_goal_path',
                        help="File path for obstacle set")
    args = parser.parse_args()

    fig, ax = plt.subplots()

    path = build_obstacle_course(args.obstacle_path, ax)
    start, goal = add_start_and_goal(args.start_goal_path, ax) # tuple (x, y)

    obstacles = read_obstacles(args.obstacle_path)

    rrt(start, goal, obstacles, plt, step_size=40, bias=0.05)

    # bi_rrt(start, goal, obstacles, plt, step_size=40, bias=0.05)

    plt.show()
