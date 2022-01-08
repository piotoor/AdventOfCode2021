import itertools
import operator
from utilities import non_blank_lines


def parse_day9_data():
    with open("day9.txt", "r") as f:
        data = [list(map(int, list(line))) for line in non_blank_lines(f)]

    return data


def is_low_point(r, c, data):
    for i in range(r - 1, r + 2):
        for j in range(c - 1, c + 2):
            if i < 0 or i >= len(data) or j < 0 or j >= len(data[0]) or (i != r and j != c) or (i == r and j == c):
                continue

            if data[i][j] <= data[r][c]:
                return False

    return True


def calculate_sum_of_the_risk_levels(data):
    ans = 0

    for r in range(len(data)):
        for c in range(len(data[0])):

            if is_low_point(r, c, data):
                ans += data[r][c] + 1

    return ans


def day9_a():
    data = parse_day9_data()
    print("day9_a = {}".format(calculate_sum_of_the_risk_levels(data)))


class BasinHandler:
    def __init__(self, data):
        self.data = data
        self.visited = [[-1 if x != 9 else x for x in row] for row in data]
        self.curr = 0
        self.next_basin_coords = (0, 0)
        self.basin_sizes = []

    def dfs(self, r, c):
        if r < 0 or c < 0 or r >= len(self.visited) or c >= len(self.visited[0]) or self.visited[r][c] in [9,
                                                                                                           self.curr]:
            return

        self.visited[r][c] = self.curr
        self.basin_sizes[self.curr] += 1
        for next_r, next_c in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            self.dfs(next_r, next_c)

    def find_next_basin(self):
        for i in range(len(self.visited)):
            for j in range(len(self.visited[0])):
                if self.visited[i][j] == -1:
                    self.next_basin_coords = (i, j)
                    return True

        return False

    def mark_next_basin(self):
        self.basin_sizes.append(0)
        r, c = self.next_basin_coords
        self.dfs(r, c)
        self.curr += 1

    def calculate_sum_of_three_largest_basins(self):
        while self.find_next_basin():
            self.mark_next_basin()

        basins_from_largest = list(reversed(sorted(self.basin_sizes)))
        return list(itertools.accumulate(basins_from_largest[:3], operator.mul))[2]


def day9_b():
    data = parse_day9_data()
    basin_handler = BasinHandler(data)
    print("day9_b = {}".format(basin_handler.calculate_sum_of_three_largest_basins()))


def day9():
    day9_a()
    day9_b()
