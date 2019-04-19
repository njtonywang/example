# usage:  python visualize_map.py obstacles_file start_goal_file

from __future__ import division
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
import random
import math
from sklearn.neighbors import NearestNeighbors
from shapely.geometry import Polygon, LineString
import copy
from dijkstar import Graph, find_path


def build_obstacle_course(obstacle_path, ax):
    polygon = []
    poly_vertices = []
    vertices = list()
    codes = [Path.MOVETO]
    with open(obstacle_path) as f:
        quantity = int(f.readline())
        lines = 0
        for line in f:
            coordinates = tuple(map(int, line.strip().split(' ')))
            if len(coordinates) == 1:
                codes += [Path.MOVETO] + [Path.LINETO]*(coordinates[0]-1) + [Path.CLOSEPOLY]
                vertices.append((0, 0))  # Always ignored by closepoly command
                if len(poly_vertices) > 0:
                    # print(poly_vertices)
                    polygon.append(Polygon(poly_vertices))
                    poly_vertices = []
            else:
                poly_vertices.append(coordinates)
                vertices.append(coordinates)

    polygon.append(Polygon(poly_vertices))
    # print(poly_vertices)
    vertices.append((0, 0))
    vertices = np.array(vertices, float)
    path = Path(vertices, codes)
    pathpatch = patches.PathPatch(path, facecolor='None', edgecolor='xkcd:violet')

    ax.add_patch(pathpatch)
    ax.set_title('Sample-Based Motion Planning')

    ax.dataLim.update_from_data_xy(vertices)
    ax.autoscale_view()
    ax.invert_yaxis()

    return path, polygon


def add_start_and_goal(start_goal_path, ax):
    start, goal = None, None
    with open(start_goal_path) as f:
        start = tuple(map(int, f.readline().strip().split(' ')))
        goal  = tuple(map(int, f.readline().strip().split(' ')))

    ax.add_patch(patches.Circle(start, facecolor='xkcd:bright green'))
    ax.add_patch(patches.Circle(goal, facecolor='xkcd:fuchsia'))

    return start, goal


def add_samples(n, ax, path):
    samples = []
    # j = 0
    i = 0
    while i < n:
        x = random.randint(0, 600)
        y = random.randint(0, 600)
        if not path.contains_points([(x, y)])[0]:
            samples.append([x, y])
            ax.add_patch(patches.Circle((x, y), 1, facecolor='xkcd:blue'))
            i += 1
        # print(path.contains_points([(x, y)])[0])
    return samples


def connect_nearest_neighbors(samples, polygon):
    vertex = []
    edges = []
    X = np.array(samples)
    nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    for nq in indices:
        for i in range(1, len(nq)):
            edge = [tuple(samples[nq[0]]), tuple(samples[nq[i]])]
            edge_reversed = [tuple(samples[nq[i]]), tuple(samples[nq[0]])]
            path = LineString(edge)
            intersect = intersect_polygon(path, polygon)
            # print(path)
            if edge not in edges and edge_reversed not in edges and not intersect:
                edges.append(edge)
                vertex.append(edge[0])
                vertex.append(edge[1])
                plt.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], linewidth=0.5, color='blue')

    return edges, vertex


def intersect_polygon(path, polygon):
    for p in polygon:
        if path.intersects(p):
            return True
    return False


def find_closest(vertex, point):
    res = tuple()
    smallest_dist = 1000000
    for v in vertex:
        a = np.array(point)
        b = np.array(v)
        dist = np.linalg.norm(a - b)
        if dist < smallest_dist:
            res = v
            smallest_dist = dist
        # print(smallest_dist)
    return res


def add_start_goal(vertex, point, polygon, edges):
    copy_vertex = copy.deepcopy(vertex)
    closest = find_closest(copy_vertex, point)
    path = LineString([closest, point])
    while intersect_polygon(path, polygon):
        copy_vertex.remove(closest)
        closest = find_closest(copy_vertex, point)
        path = LineString([closest, point])
    plt.plot([point[0], closest[0]], [point[1], closest[1]], linewidth=1, color='red')
    edges.append([point, closest])
    return edges


def find_shortest_path(edges, vertex, s_i, e_i):
    print(s_i)
    graph = Graph()
    for edge in edges:
        a = vertex.index(edge[0])
        b = vertex.index(edge[1])
        dist = np.linalg.norm(np.array(edge[0]) - np.array(edge[1]))
        print(a, b, dist)
        graph.add_edge(a, b, {'cost': dist})
        graph.add_edge(b, a, {'cost': dist})
    cost_func = lambda u, v, e, prev_e: e['cost']
    shortest = find_path(graph, s_i, e_i, cost_func=cost_func)[0]
    for i in range(len(shortest) - 1):
        point1 = vertex[shortest[i]]
        point2 = vertex[shortest[i + 1]]
        plt.plot([point1[0], point2[0]], [point1[1], point2[1]], linewidth=2, color='g')



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('obstacle_path',
                        help="File path for obstacle set")
    parser.add_argument('start_goal_path',
                        help="File path for obstacle set")
    args = parser.parse_args()

    fig, ax = plt.subplots()
    path, polygon = build_obstacle_course(args.obstacle_path, ax)
    start, goal = add_start_and_goal(args.start_goal_path, ax)

    samples = add_samples(500, ax, path)
    edges, vertex = connect_nearest_neighbors(samples, polygon)
    print(find_closest(vertex, start))
    print(find_closest(vertex, goal))
    # print(len(vertex))
    print(len(edges))
    vertex = list(set(vertex))
    edges = add_start_goal(vertex, start, polygon, edges)
    edges = add_start_goal(vertex, goal, polygon, edges)
    vertex.append(start)
    vertex.append(goal)
    print(len(vertex))
    print(vertex.index(goal))
    find_shortest_path(edges, vertex, vertex.index(start), vertex.index(goal))
    print(edges)
    # print(samples)
    # print(polygon)
    plt.show()