from utilities import non_blank_lines
import time
import heapq
import sys


class WeightedPathfinder:
    def __init__(self, data):
        self.data = data
        self.data_5x5 = WeightedPathfinder.extend_data(self.data)

    @classmethod
    def extend_data(cls, data):
        data_5x5 = [[0 for _ in range(len(data[0]) * 5)] for _ in range(len(data[0]) * 5)]

        for i in range(5):
            for r in range(len(data)):
                for j in range(5):
                    for c in range(len(data[0])):
                        new_cell = (data[r][c] - 1 + i + j) % 9 + 1
                        data_5x5[i * len(data) + r][j * len(data[0]) + c] = new_cell

        return data_5x5

    @classmethod
    def print_path(cls, data, path):
        data_str = ""
        data_ = data.copy()

        for i in range(len(data_)):
            for j in range(len(data_[0])):
                if (i, j) in path:
                    data_[i][j] = 'X'
                else:
                    data_[i][j] = '_'
            data_str += "".join(data_[i]) + '\n'

        print(data_str)

    class Node:
        def __init__(self, coords, dist):
            self.coords = coords
            self.dist = dist

        def __lt__(self, other):
            return self.dist < other.dist

    @classmethod
    def dijkstra(cls, data, source, target):
        dist = {}
        prev = {}
        neighbours = [[set() for _ in data[0]] for _ in data]
        qq = []
        heapq.heappush(qq, WeightedPathfinder.Node(source, 0))

        for r in range(len(data)):
            for c in range(len(data[0])):
                curr = (r, c)
                dist[curr] = sys.maxsize

                for i in range(max(0, r - 1), min(r + 2, len(data))):
                    for j in range(max(0, c - 1), min(c + 2, len(data[0]))):
                        if (i == r and j == c) or (i != r and j != c):
                            continue
                        neighbours[r][c].add((i, j))

        dist[source] = 0

        while len(qq) > 0:
            curr = heapq.heappop(qq)
            if curr.coords == target:
                return dist, prev

            r, c = curr.coords
            for ngh in neighbours[r][c]:
                i, j = ngh
                alt = dist[curr.coords] + data[i][j]

                if alt < dist[ngh]:
                    dist[ngh] = alt
                    prev[ngh] = curr.coords
                    heapq.heappush(qq, WeightedPathfinder.Node(ngh, dist[ngh]))

        return dist, prev

    def find_path_of_the_lowest_risk(self):
        source = (0, 0)
        target = (len(self.data) - 1, len(self.data[0]) - 1)
        dist, prev = WeightedPathfinder.dijkstra(self.data, source, target)
        return dist[target]

    def find_path_of_the_lowest_risk_extended(self):
        source = (0, 0)
        target = (len(self.data_5x5) - 1, len(self.data_5x5[0]) - 1)
        dist, prev = WeightedPathfinder.dijkstra(self.data_5x5, source, target)
        return dist[target]


def parse_day15_data():
    with open("day15.txt", "r") as f:
        data = [list(map(int, list(line))) for line in non_blank_lines(f)]

    return data


def day15_a():
    data = parse_day15_data()
    start = time.time()
    pathfinder = WeightedPathfinder(data)
    print("day15_a = {}".format(pathfinder.find_path_of_the_lowest_risk()))
    print("duration = {}".format(time.time() - start))


def day15_b():
    data = parse_day15_data()
    start = time.time()
    pathfinder = WeightedPathfinder(data)
    print("day15_b = {}".format(pathfinder.find_path_of_the_lowest_risk_extended()))
    print("duration = {}".format(time.time() - start))


def day15():
    day15_a()
    day15_b()
