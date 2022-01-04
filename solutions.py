import collections
import operator
from abc import ABC, abstractmethod
import sys
import itertools
import statistics
from collections import Counter
import time
import heapq
import copy
import math
from operator import add
import re
from math import copysign

sys.setrecursionlimit(15000)


def count_number_of_increases(data):
    ans = 0

    for i in range(1, len(data)):
        if data[i] > data[i - 1]:
            ans += 1

    return ans


def count_number_of_increased_windows(data):
    ans = 0

    if len(data) <= 3:
        return ans

    prev = sum(data[0:3])
    for i in range(3, len(data) + 1):
        curr = sum(data[i - 3:i])
        if curr > prev:
            ans += 1
        prev = curr

    return ans


def parse_day1_data():
    with open("day1a.txt", "r") as f:
        data = list(map(int, f.read().splitlines()))

    return data


def day1_a():
    data = parse_day1_data()
    print("day1_a = {}".format(count_number_of_increases(data)))


def day1_b():
    data = parse_day1_data()
    print("day1_b = {}".format(count_number_of_increased_windows(data)))


def calculate_hrz_depth_product(data):
    hrz, depth = 0, 0
    for x in data:
        cmd, arg = x

        if cmd == "forward":
            hrz += int(arg)
        elif cmd == "up":
            depth -= int(arg)
        elif cmd == "down":
            depth += int(arg)

    return hrz * depth


def parse_day2_data():
    with open("day2a.txt", "r") as f:
        data = [tuple(cmd.split()) for cmd in f.read().splitlines()]

    return data


def day2_a():
    data = parse_day2_data()
    print("day2_a = {}".format(calculate_hrz_depth_product(data)))


def calculate_hrz_depth_aim_product(data):
    hrz, depth, aim = 0, 0, 0
    for x in data:
        cmd, arg = x

        if cmd == "forward":
            hrz += int(arg)
            depth += aim * int(arg)
        elif cmd == "up":
            aim -= int(arg)
        elif cmd == "down":
            aim += int(arg)

    return hrz * depth


def day2_b():
    data = parse_day2_data()
    print("day2_b = {}".format(calculate_hrz_depth_aim_product(data)))


def calculate_power_consumption(data):
    bit_count = [0] * len(data[0])
    for x in data:
        for i in range(len(x)):
            bit_count[i] += int(x[i])

    gamma_rate_list = [1 if x >= len(data) / 2 else 0 for x in bit_count]
    epsilon_rate_list = [1 if x < len(data) / 2 else 0 for x in bit_count]

    gamma_rate = int("".join(map(str, gamma_rate_list)), 2)
    epsilon_rate = int("".join(map(str, epsilon_rate_list)), 2)
    return gamma_rate * epsilon_rate


def parse_day3_data():
    with open("day3a.txt", "r") as f:
        data = list(f.read().splitlines())

    return data


def day3_a():
    data = parse_day3_data()
    print("day3_a = {}".format(calculate_power_consumption(data)))


def calculate_rating(data, bit):
    curr_data = data.copy()

    i = 0
    while len(curr_data) > 1 and i < len(data[0]):
        bit_count_ith_column = sum(bit if x[i] == str(bit) else int(not bit) for x in curr_data)

        if bit_count_ith_column >= len(curr_data) / 2:
            curr_data = list(filter(lambda a: a[i] == str(bit), curr_data))
        else:
            curr_data = list(filter(lambda a: a[i] == str(int(not bool(bit))), curr_data))

        i += 1

    return int(curr_data[0], 2)


def calculate_oxygen_generator_rating(data):
    return calculate_rating(data, 1)


def calculate_co2_scrubber_rating(data):
    return calculate_rating(data, 0)


def calculate_life_support_rating(data):
    return calculate_oxygen_generator_rating(data) * calculate_co2_scrubber_rating(data)


def day3_b():
    data = parse_day3_data()
    print("day3_b = {}".format(calculate_life_support_rating(data)))


class BaseBingo(ABC):
    def __init__(self, boards, nums):
        self.boards = boards
        self.nums = nums
        self.boards_points = [0] * len(self.boards)
        self.boards_matches = [[[False for _ in range(5)] for _ in range(5)] for _ in range(len(boards))]
        self.boards_winners = [False for _ in self.boards]

        self.curr_num_ind = 0
        self.curr_num = -1
        self.winner_board_ind = -1

    @abstractmethod
    def is_move_possible(self):
        pass

    def board_wins(self, b):
        board = self.boards_matches[b]
        board_t = map(list, zip(*self.boards_matches[b]))

        if any([all(x) for x in board] + [all(x) for x in board_t]):
            return True

        return False

    def get_winner_points(self):
        return self.boards_points[self.winner_board_ind]

    def update_boards_matches(self, b):
        for r in range(5):
            for c in range(5):
                if self.boards[b][r][c] == self.curr_num:
                    self.boards_matches[b][r][c] = True

    def calculate_board_points(self, b):
        ans = 0

        for r in range(5):
            for c in range(5):
                if not self.boards_matches[b][r][c]:
                    ans += self.boards[b][r][c]

        ans *= self.curr_num

        self.boards_points[b] = ans

    @abstractmethod
    def is_target_board(self):
        pass

    def step(self):
        self.curr_num = self.nums[self.curr_num_ind]
        for b in range(len(self.boards)):
            self.update_boards_matches(b)

            if self.board_wins(b):
                if self.is_target_board():
                    self.winner_board_ind = b
                    self.calculate_board_points(b)
                self.boards_winners[b] = True

        self.curr_num_ind += 1


class Bingo(BaseBingo):
    def __init__(self, boards, nums):
        super().__init__(boards, nums)

    def is_move_possible(self):
        return self.curr_num_ind < len(self.nums) and not any(self.boards_winners)

    def is_target_board(self):
        return self.boards_winners.count(True) == 0


class AntiBingo(BaseBingo):
    def __init__(self, boards, nums):
        super().__init__(boards, nums)

    def is_move_possible(self):
        return self.curr_num_ind < len(self.nums) and not all(self.boards_winners)

    def is_target_board(self):
        return self.boards_winners.count(False) == 1


def non_blank_lines(f):
    for ln in f:
        line = ln.rstrip()
        if line:
            yield line


def parse_day4_data():
    raw_data = []
    with open("day4_a.txt", "r") as f:
        numbers = list(map(int, f.readline().split(",")))
        for line in non_blank_lines(f):
            raw_data.append(list(map(int, line.split())))
    data = []

    for i in range(0, len(raw_data), 5):
        board = []
        for row in range(i, i + 5):
            board.append(raw_data[row])

        data.append(board)

    return data, numbers


def day4_a():
    data, numbers = parse_day4_data()
    bingo = Bingo(data, numbers)

    while bingo.is_move_possible():
        bingo.step()

    print("day4_a = {}".format(bingo.get_winner_points()))


def day4_b():
    data, numbers = parse_day4_data()
    bingo = AntiBingo(data, numbers)

    while bingo.is_move_possible():
        bingo.step()

    print("day4_b = {}".format(bingo.get_winner_points()))


def parse_day5_data():
    with open("day5_a.txt", "r") as f:
        data = [line.split(" -> ") for line in non_blank_lines(f)]

    data = [tuple(map(int, (x[0] + "," + x[1]).split(","))) for x in data]
    return data


def draw_horizontal_vertical(diagram, data):
    horizontal_and_vertical = list(filter(lambda x: x[0] == x[2] or x[1] == x[3], data))
    for h in horizontal_and_vertical:
        x0, y0, x1, y1 = h

        if x0 == x1:
            for i in range(min(y0, y1), max(y0, y1) + 1):
                if (x0, i) in diagram:
                    diagram[(x0, i)] += 1
                else:
                    diagram[(x0, i)] = 1
        elif y0 == y1:
            for i in range(min(x0, x1), max(x0, x1) + 1):
                if (i, y0) in diagram:
                    diagram[(i, y0)] += 1
                else:
                    diagram[(i, y0)] = 1

    return diagram


def draw_diagonal(diagram, data):
    diagonal = list(filter(lambda x: abs(x[0] - x[2]) == abs(x[1] - x[3]), data))
    for d in diagonal:
        x0, y0, x1, y1 = d
        curr_x = x0
        curr_y = y0

        if x0 < x1:
            step_x = 1
        else:
            step_x = -1

        if y0 < y1:
            step_y = 1
        else:
            step_y = -1

        for i in range(abs(curr_x - x1) + 1):
            if (curr_x, curr_y) in diagram:
                diagram[(curr_x, curr_y)] += 1
            else:
                diagram[(curr_x, curr_y)] = 1

            curr_x += step_x
            curr_y += step_y

    return diagram


def count_overlapping_horizontal_vertical(data):
    diagram = {}
    diagram = draw_horizontal_vertical(diagram, data)

    vls = diagram.values()
    return len(list(filter(lambda x: x > 1, vls)))


def day5_a():
    data = parse_day5_data()
    print("day5_a = {}".format(count_overlapping_horizontal_vertical(data)))


def count_overlapping_horizontal_vertical_diagonal(data):
    diagram = {}
    diagram = draw_horizontal_vertical(diagram, data)
    diagram = draw_diagonal(diagram, data)

    vls = diagram.values()
    return len(list(filter(lambda x: x > 1, vls)))


def day5_b():
    data = parse_day5_data()
    print("day5_b = {}".format(count_overlapping_horizontal_vertical_diagonal(data)))


def parse_day6_data():
    with open("day6_a.txt", "r") as f:
        data = list(map(int, f.readline().split(",")))

    return data


def count_lanternfish(lanternfish, days):
    lanternfish_map = {}
    for x in lanternfish:
        if x in lanternfish_map:
            lanternfish_map[x] += 1
        else:
            lanternfish_map[x] = 1

    while days > 0:
        tmp_map = {}
        for k, v in lanternfish_map.items():
            if k - 1 < 0:
                if 8 in tmp_map:
                    tmp_map[8] += v
                else:
                    tmp_map[8] = v

                if 6 in tmp_map:
                    tmp_map[6] += v
                else:
                    tmp_map[6] = v
            else:
                if k - 1 in tmp_map:
                    tmp_map[k - 1] += v
                else:
                    tmp_map[k - 1] = v

        lanternfish_map = tmp_map
        days -= 1

    return sum(lanternfish_map.values())


def day6_a():
    data = parse_day6_data()
    print("day6_a = {}".format(count_lanternfish(data, 80)))


def day6_b():
    data = parse_day6_data()
    print("day6_b = {}".format(count_lanternfish(data, 256)))


def parse_day7_data():
    with open("day7_a.txt", "r") as f:
        data = list(map(int, f.readline().split(",")))

    return data


def find_optimal_fuel_usage(data, fuel_increase):
    min_x = min(data)
    max_x = max(data)

    optimal_fuel_usage = sys.maxsize

    for i in range(min_x, max_x + 1):
        curr_fuel_usage = 0

        for x in data:
            curr_fuel_usage += fuel_increase(abs(i - x))

        optimal_fuel_usage = min(optimal_fuel_usage, curr_fuel_usage)

    return optimal_fuel_usage


def day7_a():
    data = parse_day7_data()
    print("day7_a = {}".format(find_optimal_fuel_usage(data, lambda x: x)))


def day7_b():
    data = parse_day7_data()
    print("day7_b = {}".format(find_optimal_fuel_usage(data, lambda x: int(0.5 * x * (x + 1)))))


def parse_day8_data():
    with open("day8_a.txt", "r") as f:
        data = [[x.split() for x in line.split(" | ")] for line in non_blank_lines(f)]

    return data


def count_digits_made_of_unique_number_of_segments(data):
    number_of_digits_of_unique_seqments = 0

    for x in data:
        output = x[1]
        # print("output = {}, filtered = {}".format(output, list(filter(lambda y: len(y) in [2, 3, 4, 7], output))))
        number_of_digits_of_unique_seqments += len(list(filter(lambda y: len(y) in [2, 3, 4, 7], output)))

    return number_of_digits_of_unique_seqments


def day8_a():
    data = parse_day8_data()
    print("day8_a = {}".format(count_digits_made_of_unique_number_of_segments(data)))


def calculate_sum_of_all_output_digits(data):
    ans = 0

    for line in data:
        signal_patterns = line[0]
        identified_digits = [set()] * 10
        remaining_digits = []
        segments = [set()] * 7

        for digit in signal_patterns:
            if len(digit) == 2:
                identified_digits[1] = set(digit)
            elif len(digit) == 4:
                identified_digits[4] = set(digit)
            elif len(digit) == 3:
                identified_digits[7] = set(digit)
            elif len(digit) == 7:
                identified_digits[8] = set(digit)
            else:
                remaining_digits.append(set(digit))

        segments[0] = identified_digits[7] - identified_digits[1]

        for x in remaining_digits:  # 9, 6, 0
            if len(x) == 6 and len(x - identified_digits[7] - identified_digits[4]) == 1:
                segments[6] = x - identified_digits[7] - identified_digits[4]
                break

        segments[4] = identified_digits[8] - segments[0] - segments[6] - identified_digits[4]

        for x in remaining_digits:  # 5, 3, 2
            if len(x) == 5 and len(x - segments[0] - segments[4] - segments[6] - identified_digits[1]) == 1:
                segments[3] = x - segments[0] - segments[4] - segments[6] - identified_digits[1]
                break

        segments[1] = identified_digits[4] - segments[3] - identified_digits[1]

        for x in remaining_digits:  # 9, 6, 0
            if len(x) == 6 and len(x - segments[0] - segments[1] - segments[3] - segments[4] - segments[6]) == 1:
                segments[5] = x - segments[0] - segments[1] - segments[3] - segments[4] - segments[6]
                break

        all_segments = set()
        for x in segments:
            if x is not None:
                all_segments = all_segments | x

        segments[2] = identified_digits[8] - all_segments

        identified_digits[0] = identified_digits[8] - segments[3]
        identified_digits[2] = identified_digits[8] - segments[1] - segments[5]
        identified_digits[3] = identified_digits[8] - segments[1] - segments[4]
        identified_digits[5] = identified_digits[8] - segments[2] - segments[4]
        identified_digits[6] = identified_digits[8] - segments[2]
        identified_digits[9] = identified_digits[8] - segments[4]

        identified_digits_dict = {}
        for i in range(len(identified_digits)):
            identified_digits_dict["".join(sorted(identified_digits[i]))] = i

        output_num = ""
        for output_digs in line[1]:
            output_num = output_num + str(identified_digits_dict["".join(sorted(output_digs))])

        ans += int(output_num)

    return ans


def day8_b():
    data = parse_day8_data()
    print("day8_a = {}".format(calculate_sum_of_all_output_digits(data)))


def parse_day9_data():
    with open("day9_a.txt", "r") as f:
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


class ParenthesisParser:
    brackets_pairs = {'[': ']', '(': ')', '<': '>', '{': '}'}
    opening_brackets = {'[', '(', '<', '{'}
    syntax_error_points = {
        ')': 3,
        ']': 57,
        '}': 1197,
        '>': 25137,
    }
    autocomplete_points = {
        ')': 1,
        ']': 2,
        '}': 3,
        '>': 4,
    }

    def __init__(self, data):
        self.data = data
        self.invalid_brackets = []
        self.missing_brackets = []

    @classmethod
    def find_first_corrupted_bracket_in_expr(cls, expr):
        stack = []

        for x in expr:
            if x in ParenthesisParser.opening_brackets:
                stack.append(x)
            elif ParenthesisParser.brackets_pairs[stack[-1]] == x:
                stack.pop()
            else:
                return x

        return None

    def find_invalid_brackets(self):
        for row in self.data:
            first_corrupted = ParenthesisParser.find_first_corrupted_bracket_in_expr(row)
            if first_corrupted is not None:
                self.invalid_brackets.append(first_corrupted)

    def calculate_syntax_error_score(self):
        self.find_invalid_brackets()
        ans = 0

        for x in self.invalid_brackets:
            ans += ParenthesisParser.syntax_error_points[x]

        return ans

    @classmethod
    def find_missing_brackets_in_expr(cls, expr):
        stack = []

        for x in expr:
            if x in ParenthesisParser.opening_brackets:
                stack.append(x)
            elif ParenthesisParser.brackets_pairs[stack[-1]] == x:
                stack.pop()
            else:
                return None

        return "".join(reversed(list(map(lambda y: ParenthesisParser.brackets_pairs[y], stack))))

    def find_missing_brackets(self):
        for row in self.data:
            missing_brackets = ParenthesisParser.find_missing_brackets_in_expr(row)
            if missing_brackets is not None:
                self.missing_brackets.append(missing_brackets)

    def calculate_autocomplete_score(self):
        self.find_missing_brackets()
        scores = [0] * len(self.missing_brackets)

        for i in range(len(self.missing_brackets)):
            for x in self.missing_brackets[i]:
                scores[i] *= 5
                scores[i] += ParenthesisParser.autocomplete_points[x]

        return statistics.median(scores)


def parse_day10_data():
    with open("day10_a.txt", "r") as f:
        data = [line for line in non_blank_lines(f)]

    return data


def day10_a():
    data = parse_day10_data()
    parser = ParenthesisParser(data)

    print("day10_a = {}".format(parser.calculate_syntax_error_score()))


def day10_b():
    data = parse_day10_data()
    parser = ParenthesisParser(data)

    print("day10_b = {}".format(parser.calculate_autocomplete_score()))


def parse_day11_data():
    with open("day11_a.txt", "r") as f:
        data = [list(map(int, list(line))) for line in non_blank_lines(f)]

    return data


class OctopusEngeryHandler:
    def __init__(self, data):
        self.data = data
        self.num_of_flashes = 0
        self.already_flashed = set()

    def increment_energy_levels(self):
        for i in range(len(self.data)):
            self.data[i] = [x + 1 for x in self.data[i]]

    def dfs(self, r, c):
        if r < 0 or c < 0 or r >= len(self.data) or c >= len(self.data[0]) or (r, c) in self.already_flashed:
            return

        if self.data[r][c] == 10:
            self.already_flashed.add((r, c))
            self.num_of_flashes += 1
            for i in range(r - 1, r + 2):
                for j in range(c - 1, c + 2):
                    self.dfs(i, j)
        else:
            self.data[r][c] += 1
            if self.data[r][c] == 10:
                self.dfs(r, c)

    def propagate_energy_levels(self):
        for i in range(len(self.data)):
            for j in range(len(self.data[0])):
                if self.data[i][j] == 10:
                    self.dfs(i, j)

    def reset_energy_levels(self):
        for i in range(len(self.data)):
            self.data[i] = [0 if x > 9 else x for x in self.data[i]]

    def count_number_of_flashes(self, num_of_steps):
        for i in range(num_of_steps):
            self.increment_energy_levels()
            self.propagate_energy_levels()
            self.reset_energy_levels()
            self.already_flashed.clear()

        return self.num_of_flashes

    def count_steps_to_simultaneous_flash(self):
        step = 0

        while len(self.already_flashed) < len(self.data) * len(self.data[0]):
            self.already_flashed.clear()
            self.increment_energy_levels()
            self.propagate_energy_levels()
            self.reset_energy_levels()

            step += 1

        return step


def day11_a():
    data = parse_day11_data()
    num_of_steps = 100
    handler = OctopusEngeryHandler(data)
    print("day11_a = {}".format(handler.count_number_of_flashes(num_of_steps)))


def day11_b():
    data = parse_day11_data()
    handler = OctopusEngeryHandler(data)
    print("day11_b = {}".format(handler.count_steps_to_simultaneous_flash()))


class Pathfinder:
    def __init__(self, data):
        self.data = data
        self.adjacency_dict = {}
        self.data_to_adjacency_dict()

        self.visited = set()
        self.curr_path = []
        self.paths = []
        self.small_nodes_allowed_visits = {}
        self.reset_small_nodes_allowed_visits()

    def data_to_adjacency_dict(self):
        for x, y in self.data:
            if x in self.adjacency_dict:
                self.adjacency_dict[x].add(y)
            else:
                self.adjacency_dict[x] = {y}
            if y in self.adjacency_dict:
                self.adjacency_dict[y].add(x)
            else:
                self.adjacency_dict[y] = {x}

    def dfs(self, node):
        if node == "end":
            self.paths.append(self.curr_path.copy())
        else:
            for n in self.adjacency_dict[node]:
                if n in self.small_nodes_allowed_visits.keys() and self.small_nodes_allowed_visits[n] == 0:
                    continue

                self.curr_path.append(n)
                if n.islower():
                    self.small_nodes_allowed_visits[n] -= 1

                self.dfs(n)

                if n.islower():
                    self.small_nodes_allowed_visits[n] += 1
                self.curr_path.pop()

    def count_paths_visiting_all_small_caves_once(self):
        self.paths = []
        self.reset_small_nodes_allowed_visits()
        self.curr_path = ["start"]
        self.small_nodes_allowed_visits["start"] = 0
        self.dfs("start")

        return len(self.paths)

    def reset_small_nodes_allowed_visits(self):
        self.small_nodes_allowed_visits = {x: 1 for x in set(filter(lambda x: x.islower(), self.adjacency_dict))}

    def count_paths_visiting_all_small_caves_but_one_once(self):
        self.paths = []

        for small in self.small_nodes_allowed_visits:
            self.reset_small_nodes_allowed_visits()
            self.small_nodes_allowed_visits[small] = 2
            self.curr_path = ["start"]
            self.small_nodes_allowed_visits["start"] = 0
            self.dfs("start")

        return len(set(tuple(p) for p in self.paths))


def parse_day12_data():
    with open("day12_a.txt", "r") as f:
        data = [tuple(line.split("-")) for line in non_blank_lines(f)]

    return data


def day12_a():
    data = parse_day12_data()
    pf = Pathfinder(data)
    print("day12_a = {}".format(pf.count_paths_visiting_all_small_caves_once()))


def day12_b():
    data = parse_day12_data()
    pf = Pathfinder(data)
    print("day12_b = {}".format(pf.count_paths_visiting_all_small_caves_but_one_once()))


def parse_day13_data():
    dots = set()
    instructions = []
    with open("day13_a.txt", "r") as f:
        for line in non_blank_lines(f):
            if line[0] == 'f':
                fold_axis = line.split(" ")[2].split("=")
                instructions.append((fold_axis[0], int(fold_axis[1])))

            else:
                dots.add(tuple(map(int, line.split(","))))

    return dots, instructions


class OrigamiFolder:
    def __init__(self, data):
        self.dots = data[0].copy()
        self.instructions = data[1].copy()

    def print(self):
        max_x = max([x[0] for x in self.dots])
        max_y = max([x[1] for x in self.dots])
        paper = [['.'] * (max_x + 1) for _ in range(max_y + 1)]

        for x, y in self.dots:
            paper[y][x] = '#'

        for r in paper:
            print(r)

        print("\n")

    def fold_left(self, axis):
        new_dots = set()
        for dot in self.dots:
            x, y = dot
            if x > axis:
                dot_prime = (axis - abs(axis - x), y)
                new_dots.add(dot_prime)
            else:
                new_dots.add(dot)

        self.dots = new_dots

    def fold_up(self, axis):
        new_dots = set()
        for dot in self.dots:
            x, y = dot
            if y > axis:
                dot_prime = (x, axis - abs(axis - y))
                new_dots.add(dot_prime)
            else:
                new_dots.add(dot)

        self.dots = new_dots

    def fold(self, axis_label, axis):
        if axis_label == 'x':
            self.fold_left(axis)
        elif axis_label == 'y':
            self.fold_up(axis)

    def execute_instructions(self, steps):
        # self.print()
        for i in range(min(steps, len(self.instructions))):
            axis, val = self.instructions[i]
            self.fold(axis, val)
            # self.print()

    def count_dots_after_single_fold(self):
        self.execute_instructions(1)

        return len(self.dots)

    def count_dots_after_full_fold(self):
        self.execute_instructions(len(self.instructions))

        return len(self.dots)


def day13_a():
    data = parse_day13_data()
    folder = OrigamiFolder(data)
    print("day13_a = {}".format(folder.count_dots_after_single_fold()))


def day13_b():
    data = parse_day13_data()
    folder = OrigamiFolder(data)
    print("day13_a = {}".format(folder.count_dots_after_full_fold()))
    folder.print()


def parse_day14_data():
    data = []

    with open("day14_a.txt", "r") as f:
        data.append(f.readline().rstrip())
        for line in non_blank_lines(f):
            data.append(tuple(line.split(" -> ")))

    return data


class PolymerHandler:
    def __init__(self, data):
        self.polymer = data[0]
        self.instructions = data[1:]
        self.instructions_dict = {pair: to_insert for pair, to_insert in self.instructions}
        self.cache = {}

    @classmethod
    def compute_char_freq_dict(cls, s):
        char_freq = {}

        for x in s:
            if x in char_freq:
                char_freq[x] += 1
            else:
                char_freq[x] = 1

        return char_freq

    def execute_instructions(self, polymer_piece, steps):
        if steps == 0:
            return PolymerHandler.compute_char_freq_dict(polymer_piece[:-1])
        else:
            cache_entry = (polymer_piece, steps)

            if cache_entry not in self.cache:
                self.cache[cache_entry] = dict()
                for x in [polymer_piece[i: i + 2] for i in range(len(polymer_piece) - 1)]:
                    char_freq = self.execute_instructions(x[0] + self.instructions_dict[x] + x[1], steps - 1)
                    self.cache[cache_entry] = dict(Counter(char_freq) + Counter(self.cache[cache_entry]))

            return self.cache[cache_entry]

    def most_common_least_common_diff(self, steps):
        self.cache.clear()

        char_freq = self.execute_instructions(self.polymer, steps)
        char_freq[self.polymer[-1]] += 1
        most_common = max(char_freq.values())
        least_common = min(char_freq.values())

        return most_common - least_common


def day14_a():
    data = parse_day14_data()
    poly_handler = PolymerHandler(data)
    print("day14_a = {}".format(poly_handler.most_common_least_common_diff(10)))


def day14_b():
    data = parse_day14_data()
    poly_handler = PolymerHandler(data)
    print("day14_b = {}".format(poly_handler.most_common_least_common_diff(40)))


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
    with open("day15_a.txt", "r") as f:
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


class TransmissionHandler:
    def __init__(self, data):
        self.data_hex = data
        self.data_bin = "".join([format(int(digit, 16), '04b') for digit in self.data_hex])
        self.ind = 0
        self.sum_of_version_numbers = 0

    def parse_packet(self, data):
        ver_int = int(data[self.ind:self.ind+3], 2)
        self.ind += 3
        type_int = int(data[self.ind:self.ind+3], 2)
        self.ind += 3
        self.sum_of_version_numbers += ver_int

        if type_int == 4:
            literal = ""
            while True:
                more = data[self.ind:self.ind+1]
                self.ind += 1
                literal += data[self.ind:self.ind+4]
                self.ind += 4
                if more == '0':
                    break
            return int(literal, 2)
        else:
            length_type_id_str = data[self.ind:self.ind+1]
            self.ind += 1

            if length_type_id_str == '0':
                length_field_size = 15
            else:
                length_field_size = 11

            length_str = data[self.ind:self.ind+length_field_size]
            self.ind += length_field_size
            length_int = int(length_str, 2)

            operands_start_ind = self.ind
            operands = []
            if length_field_size == 15:
                while self.ind < operands_start_ind + length_int:
                    operands.append(self.parse_packet(data))
            else:
                for i in range(length_int):
                    operands.append(self.parse_packet(data))

        return TransmissionHandler.operator(type_int, operands)

    @classmethod
    def operator(cls, type_id, operands):
        if type_id == 0:
            return sum(operands)
        if type_id == 1:
            return math.prod(operands)
        if type_id == 2:
            return min(operands)
        if type_id == 3:
            return max(operands)
        if type_id == 5:
            return 1 if operands[0] > operands[1] else 0
        if type_id == 6:
            return 1 if operands[0] < operands[1] else 0
        if type_id == 7:
            return 1 if operands[0] == operands[1] else 0

    def calculate_sum_of_packet_versions(self):
        self.ind = 0
        self.sum_of_version_numbers = 0
        self.parse_packet(self.data_bin)
        return self.sum_of_version_numbers

    def calculate_value_of_the_outermost_packet(self):
        self.ind = 0
        return self.parse_packet(self.data_bin)


def parse_day16_data():
    with open("day16_a.txt", "r") as f:
        data = f.readline()
    return data


def day16_a():
    data = parse_day16_data()
    handler = TransmissionHandler(data)
    print("day16_a = {}".format(handler.calculate_sum_of_packet_versions()))


def day16_b():
    data = parse_day16_data()
    handler = TransmissionHandler(data)
    print("day16_b = {}".format(handler.calculate_value_of_the_outermost_packet()))


def parse_day17_data():
    data = []
    with open("day17_a.txt", "r") as f:
        line = f.readline()[13:].split(", ")
        for coords in line:
            data.extend((coords[2:].split("..")))

    return tuple(int(x) for x in data)


class ProbeLauncher:
    def __init__(self, data):
        self.target_area = data
        self.highest_y_values = []
        self.initial_velocities = set()

    def cleanup(self):
        self.highest_y_values.clear()
        self.initial_velocities.clear()

    def compute(self):
        self.cleanup()
        t_x0, t_x1, t_y0, t_y1 = self.target_area
        start_x, start_y = 0, 0
        max_v_y = 150

        for vx_init in range(0, t_x1 + 1):
            for vy_init in range(-max_v_y, max_v_y):   # todo upper
                v_x = vx_init
                v_y = vy_init
                pos_x = start_x
                pos_y = start_y
                max_y = pos_y

                while pos_x <= t_x1 and pos_y >= t_y0:
                    pos_x += v_x
                    pos_y += v_y
                    max_y = max(max_y, pos_y)

                    if t_x0 <= pos_x <= t_x1 and t_y0 <= pos_y <= t_y1:
                        self.highest_y_values.append(max_y)
                        self.initial_velocities.add((vx_init, vy_init))
                        break

                    if v_x > 0:
                        v_x -= 1
                    v_y -= 1

    def find_highest_y(self):
        return max(self.highest_y_values)

    def count_initial_velocities(self):
        return len(self.initial_velocities)


def day17_a():
    data = parse_day17_data()
    launcher = ProbeLauncher(data)
    launcher.compute()
    print("day17_a = {}".format(launcher.find_highest_y()))


def day17_b():
    data = parse_day17_data()
    launcher = ProbeLauncher(data)
    launcher.compute()
    print("day17_b = {}".format(launcher.count_initial_velocities()))


class SnailfishCalculator:
    def __init__(self, data):
        self.data = data

    @classmethod
    def explode(cls, number):
        new_number = copy.deepcopy(number)
        to_explode_indices = []
        to_update_indices = [-1, -1]

        for i in range(len(new_number)):
            if len(to_explode_indices) < 2:
                if new_number[i][1] == 4:
                    to_explode_indices.append(i)
                else:
                    to_update_indices[0] = i
            elif to_update_indices[1] == -1:
                to_update_indices[1] = i

        if len(to_explode_indices) < 2:
            return False, new_number
        if to_update_indices[0] != -1:
            new_number[to_update_indices[0]][0] += new_number[to_explode_indices[0]][0]

        if to_update_indices[1] != -1:
            new_number[to_update_indices[1]][0] += new_number[to_explode_indices[1]][0]

        new_number[to_explode_indices[0]] = [0, 3]
        del new_number[to_explode_indices[1]]

        return True, new_number

    @classmethod
    def split(cls, number):
        new_number = copy.deepcopy(number)

        for i in range(len(new_number)):
            curr_val = new_number[i][0]
            curr_depth = new_number[i][1]

            if curr_val > 9:
                left = int(math.floor(curr_val / 2.0))
                right = int(math.ceil(curr_val / 2.0))
                new_number[i] = [left, curr_depth + 1]
                new_number.insert(i + 1, [right, curr_depth + 1])
                return True, new_number

        return False, new_number

    @classmethod
    def increase_depth(cls, number):
        new_number = copy.deepcopy(number)

        for i in range(len(new_number)):
            new_number[i][1] += 1

        return new_number

    @classmethod
    def add(cls, n1, n2):
        ans = copy.deepcopy(n1)
        ans.extend(n2)
        ans = SnailfishCalculator.increase_depth(ans)

        return ans

    @classmethod
    def chain_add(cls, numbers):
        ans = None

        for x in numbers:
            if ans is None:
                ans = x
                continue

            ans = SnailfishCalculator.add(ans, x)
            reduce = True
            splitted = False
            while reduce:
                exploded, ans = SnailfishCalculator.explode(ans)
                if not exploded:
                    splitted, ans = SnailfishCalculator.split(ans)

                reduce = exploded or splitted

        return ans

    @classmethod
    def calculate_magnitude(cls, number):
        stack = []

        for x in number:
            curr = x

            while len(stack) > 0 and stack[-1][1] == curr[1]:
                curr = [3 * stack[-1][0] + 2 * curr[0], curr[1] - 1]
                stack.pop()

            stack.append(curr)

        return stack[-1][0]

    def calculate_total_magnitude(self):
        final_number = SnailfishCalculator.chain_add(self.data)
        return SnailfishCalculator.calculate_magnitude(final_number)

    def calculate_largest_sum_of_two_numbers(self):
        max_sum = 0
        for a in self.data:
            for b in self.data:
                if a != b:
                    a_b = SnailfishCalculator.chain_add([a, b])
                    b_a = SnailfishCalculator.chain_add([b, a])
                    a_b_magnitude = SnailfishCalculator.calculate_magnitude(a_b)
                    b_a_magnitude = SnailfishCalculator.calculate_magnitude(b_a)
                    max_sum = max(max_sum, a_b_magnitude, b_a_magnitude)

        return max_sum


def parse_day18_data():
    with open("day18_a.txt", "r") as f:
        raw_data = [line for line in non_blank_lines(f)]

    ans = []

    for line in raw_data:
        converted_line = []
        depth = -1
        for x in line:
            if x == '[':
                depth += 1
            elif x == ']':
                depth -= 1
            elif x.isdigit():
                converted_line.append([int(x), depth])
        ans.append(converted_line)

    return ans


def day18_a():
    data = parse_day18_data()
    calc = SnailfishCalculator(data)
    print("day18_a = {}".format(calc.calculate_total_magnitude()))


def day18_b():
    data = parse_day18_data()
    calc = SnailfishCalculator(data)
    print("day18_b = {}".format(calc.calculate_largest_sum_of_two_numbers()))


class BeaconHandler:
    def __init__(self, data):
        self.scanners = data
        self.good_scanners = {0: self.scanners[0]}
        self.scanners_positions = {0: (0, 0, 0)}

    @classmethod
    def x1(cls, point):
        x, y, z = point
        return [x, z, -y]

    @classmethod
    def x2(cls, point):
        x, y, z = point
        return [x, -y, -z]

    @classmethod
    def x3(cls, point):
        x, y, z = point
        return [x, -z, y]

    @classmethod
    def y1(cls, point):
        x, y, z = point
        return [-z, y, x]

    @classmethod
    def y2(cls, point):
        x, y, z = point
        return [-x, y, -z]

    @classmethod
    def y3(cls, point):
        x, y, z = point
        return [z, y, -x]

    @classmethod
    def z1(cls, point):
        x, y, z = point
        return [y, -x, z]

    @classmethod
    def z2(cls, point):
        x, y, z = point
        return [-x, -y, z]

    @classmethod
    def z3(cls, point):
        x, y, z = point
        return [-y, x, z]

    @classmethod
    def rotate(cls, point, rot):
        for r in rot:
            if r > 8 or r < 0:
                continue
            point = BeaconHandler.rotations_table[r](point)
        return point

    @classmethod
    def generate_all_rotations(cls, points):
        first_rot = [-1, 0, 1, 2, 6, 8]
        second_rot = [-1, 3, 4, 5]

        ans = {}

        for f in first_rot:
            for s in second_rot:
                rot = (f, s)
                ans[rot] = []
                for p in points:
                    ans[rot].append(BeaconHandler.rotate(p, rot))

        return ans

    @classmethod
    def calculate_manhattan_distance(cls, point_a, point_b):
        a_x, a_y, a_z = point_a
        b_x, b_y, b_z = point_b

        return sum((abs(a_x - b_x), abs(a_y - b_y), abs(a_z - b_z)))

    def do_scanners_overlap(self, good_ind, bad_ind):
        bad_scanner_all_rotations = BeaconHandler.generate_all_rotations(self.scanners[bad_ind])

        for rot in bad_scanner_all_rotations:
            diffs = {}
            for rotated_point in bad_scanner_all_rotations[rot]:
                for adjusted_point in self.good_scanners[good_ind]:
                    a_x, a_y, a_z = adjusted_point
                    r_x, r_y, r_z = rotated_point
                    diff = (a_x - r_x, a_y - r_y, a_z - r_z)
                    if diff in diffs:
                        diffs[diff] += 1
                    else:
                        diffs[diff] = 1

            most_freq_diff = max(diffs, key=diffs.get)
            max_diff_val = max(diffs.values())
            if max_diff_val >= 12:
                return most_freq_diff, bad_scanner_all_rotations[rot]

        return None, None

    @classmethod
    def adjust_scanner_position(cls, scanner, diff):
        adjusted_scanner = copy.deepcopy(scanner)

        for i in range(len(scanner)):
            adjusted_scanner[i][0] += diff[0]
            adjusted_scanner[i][1] += diff[1]
            adjusted_scanner[i][2] += diff[2]

        return adjusted_scanner

    def compute_scanners_positions(self):
        while len(self.scanners_positions) < len(self.scanners):
            for i in range(len(self.scanners)):
                if i in self.scanners_positions:
                    continue

                for j in self.scanners_positions:
                    diff, rotated_scanner = self.do_scanners_overlap(j, i)
                    if diff is None and rotated_scanner is None:
                        continue

                    self.good_scanners[i] = BeaconHandler.adjust_scanner_position(rotated_scanner, diff)
                    self.scanners_positions[i] = diff
                    break

    def count_beacons(self):
        self.compute_scanners_positions()
        beacons = set()
        for _, s in self.good_scanners.items():
            for b in s:
                beacons.add(tuple(b))

        return len(beacons)

    def find_largest_manhattan_distance(self):
        self.compute_scanners_positions()
        ans = 0

        for a in self.scanners_positions:
            for b in self.scanners_positions:
                if a == b:
                    continue

                pos_a = self.scanners_positions[a]
                pos_b = self.scanners_positions[b]
                manhattan_dist_a_b = BeaconHandler.calculate_manhattan_distance(pos_a, pos_b)
                ans = max(ans, manhattan_dist_a_b)

        return ans


BeaconHandler.rotations_table = [
    BeaconHandler.x1,
    BeaconHandler.x2,
    BeaconHandler.x3,

    BeaconHandler.y1,
    BeaconHandler.y2,
    BeaconHandler.y3,

    BeaconHandler.z1,
    BeaconHandler.z2,
    BeaconHandler.z3,
]


def parse_day19_data():
    with open("day19_a.txt", "r") as f:
        data = []
        for line in non_blank_lines(f):
            if line[0:3] == '---':
                data.append([])
            else:
                data[-1].append(list(map(int, line.split(","))))

    return data


def day19_a():
    data = parse_day19_data()
    handler = BeaconHandler(data)
    print("day19_a = {}".format(handler.count_beacons()))


def day19_b():
    data = parse_day19_data()
    handler = BeaconHandler(data)
    print("day19_b = {}".format(handler.find_largest_manhattan_distance()))


class ImageEnhancer:
    def __init__(self, data):
        self.alg, self.img = data

    @classmethod
    def apply_padding(cls, img, padding_char, padding_size):
        img_cols = len(img[0])
        padded_img = []

        for i in range(padding_size):
            padded_img.append([padding_char] * (img_cols + padding_size * 2))
        for row in img:
            padded_img.append([padding_char] * padding_size + row + [padding_char] * padding_size)
        for i in range(padding_size):
            padded_img.append([padding_char] * (img_cols + padding_size * 2))

        return padded_img

    def enhance_image(self, iterations):
        for i in range(iterations):
            target_img = copy.deepcopy(self.img)
            target_img = ImageEnhancer.apply_padding(target_img, '0', 1)

            if i % 2 == 0:
                self.img = ImageEnhancer.apply_padding(self.img, '0', 2)
                target_img = ImageEnhancer.apply_padding(target_img, '1', 1)
            else:
                self.img = ImageEnhancer.apply_padding(self.img, '1', 1)

            for r in range(1, len(target_img) - 1):
                for c in range(1, len(target_img[0]) - 1):
                    binary = ""
                    for rr in range(r - 1, r + 2):
                        for cc in range(c - 1, c + 2):
                            binary += self.img[rr][cc]

                    alg_ind = int(binary, 2)
                    if self.alg[alg_ind] == '#':
                        target_img[r][c] = '1'
                    else:
                        target_img[r][c] = '0'

            self.img = copy.deepcopy(target_img)

    def count_lit_pixels(self, iterations):
        self.enhance_image(iterations)
        c = collections.Counter()
        for row in self.img:
            c.update(row)

        return c['1']


def parse_day20_data():
    with open("day20_a.txt", "r") as f:
        data = [line for line in non_blank_lines(f)]

    alg = data[0]
    img = [list(map(lambda x: '1' if x == '#' else '0', line)) for line in data[1:]]
    return alg, img


def day20_a():
    data = parse_day20_data()
    enhancer = ImageEnhancer(data)
    print("day20_a = {}".format(enhancer.count_lit_pixels(2)))


def day20_b():
    data = parse_day20_data()
    enhancer = ImageEnhancer(data)
    print("day20_b = {}".format(enhancer.count_lit_pixels(50)))


class DiceRoller:
    def __init__(self, data):
        self.data = data
        self.total_universes = 0
        self.universes = [0, 0]
        self.p = [0, 0, 0, 1, 3, 6, 7, 6, 3, 1]
        self.cache = {}

    def calculate_losing_score_x_num_of_rolls(self, dice_size):
        players_pos = list(self.data)
        players_score = [0, 0]

        rolled = 1
        player = 0
        num_of_rolls = 0
        while not any(x >= 1000 for x in players_score):
            total_rolled = 0
            for i in range(3):
                num_of_rolls += 1
                total_rolled += rolled
                rolled = rolled % dice_size + 1

            players_pos[player] = (players_pos[player] + total_rolled - 1) % 10 + 1
            players_score[player] += players_pos[player]
            player = (player + 1) % 2

        return min(players_score) * num_of_rolls

    def dfs(self, players_score, players_pos, num_of_universes, player):
        if players_score[0] >= 21 or players_score[1] >= 21:
            ans = [0, 0]
            ans[player] = num_of_universes
            return ans

        ans = [0, 0]
        for total_rolled in range(3, 10):
            cache_key = (player, tuple(players_pos), tuple(players_score), total_rolled)
            if cache_key not in self.cache:
                new_players_pos = copy.deepcopy(players_pos)
                new_players_pos[player] = (new_players_pos[player] + total_rolled - 1) % 10 + 1
                new_players_score = copy.deepcopy(players_score)
                new_players_score[player] += new_players_pos[player]
                new_num_of_universes = self.p[total_rolled] * num_of_universes

                curr = self.dfs(new_players_score, new_players_pos, 1, (player + 1) % 2)
                self.cache[cache_key] = [x * new_num_of_universes for x in curr]

            ans = list(map(add, ans, self.cache[cache_key]))
        return ans

    def calculate_number_of_universes(self):
        players_pos = list(self.data)
        players_score = [0, 0]
        player = 0
        num_of_universes = 1
        ans = self.dfs(copy.deepcopy(players_score), copy.deepcopy(players_pos), num_of_universes, player)
        return max(ans)


def parse_day21_data():
    with open("day21_a.txt", "r") as f:
        data = [int(line.split(" ")[-1]) for line in non_blank_lines(f)]

    return data


def day21_a():
    data = parse_day21_data()
    roller = DiceRoller(data)
    dice_size = 100
    print("day21_a = {}".format(roller.calculate_losing_score_x_num_of_rolls(dice_size)))


def day21_b():
    data = parse_day21_data()
    roller = DiceRoller(data)
    print("day21_b = {}".format(roller.calculate_number_of_universes()))


class ReactorHandler:
    def __init__(self, data):
        self.data = data

    @classmethod
    def cuboids_overlap(cls, a, b):
        a_minx, a_maxx, a_miny, a_maxy, a_minz, a_maxz = a
        b_minx, b_maxx, b_miny, b_maxy, b_minz, b_maxz = b

        if b_minx > a_maxx or a_minx > b_maxx or b_miny > a_maxy or a_miny > b_maxy or b_minz > a_maxz or a_minz > b_maxz:
            return True
        else:
            return False

    @classmethod
    def cuboid_diff(cls, a, b):
        ans = []
        a_minx, a_maxx, a_miny, a_maxy, a_minz, a_maxz = a
        b_minx, b_maxx, b_miny, b_maxy, b_minz, b_maxz = b

        if ReactorHandler.cuboids_overlap(a, b):
            return [a]

        if a_minx < b_minx:  # L exists
            l_minx = min(a_minx, b_minx)
            l_maxx = max(a_minx, b_minx)
            l_miny = a_miny
            l_maxy = a_maxy
            l_minz = a_minz
            l_maxz = a_maxz

            ans.append((l_minx, l_maxx, l_miny, l_maxy, l_minz, l_maxz))
            if ans[-1] == a:
                return ans

        if a_maxx > b_maxx:  # R exists
            r_minx = min(a_maxx, b_maxx)
            r_maxx = max(a_maxx, b_maxx)
            r_miny = a_miny
            r_maxy = a_maxy
            r_minz = a_minz
            r_maxz = a_maxz

            ans.append((r_minx, r_maxx, r_miny, r_maxy, r_minz, r_maxz))
            if ans[-1] == a:
                return ans

        if a_maxz > b_maxz:  # F exists
            f_minx = max(a_minx, b_minx)
            f_maxx = min(a_maxx, b_maxx)
            f_miny = a_miny
            f_maxy = a_maxy
            f_minz = min(a_maxz, b_maxz)
            f_maxz = max(a_maxz, b_maxz)

            ans.append((f_minx, f_maxx, f_miny, f_maxy, f_minz, f_maxz))
            if ans[-1] == a:
                return ans

        if a_minz < b_minz:  # B exists
            bb_minx = max(a_minx, b_minx)
            bb_maxx = min(a_maxx, b_maxx)
            bb_miny = a_miny
            bb_maxy = a_maxy
            bb_minz = min(a_minz, b_minz)
            bb_maxz = max(a_minz, b_minz)

            ans.append((bb_minx, bb_maxx, bb_miny, bb_maxy, bb_minz, bb_maxz))
            if ans[-1] == a:
                return ans

        if a_miny < b_miny:  # D exists
            d_minx = max(a_minx, b_minx)
            d_maxx = min(a_maxx, b_maxx)
            d_miny = min(a_miny, b_miny)
            d_maxy = max(a_miny, b_miny)
            d_minz = max(a_minz, b_minz)
            d_maxz = min(a_maxz, b_maxz)

            ans.append((d_minx, d_maxx, d_miny, d_maxy, d_minz, d_maxz))
            if ans[-1] == a:
                return ans

        if a_maxy > b_maxy:  # U exists
            u_minx = max(a_minx, b_minx)
            u_maxx = min(a_maxx, b_maxx)
            u_miny = min(a_maxy, b_maxy)
            u_maxy = max(a_maxy, b_maxy)
            u_minz = max(a_minz, b_minz)
            u_maxz = min(a_maxz, b_maxz)

            ans.append((u_minx, u_maxx, u_miny, u_maxy, u_minz, u_maxz))
            if ans[-1] == a:
                return ans

        return ans

    @classmethod
    def cuboid_volume(cls, cuboid):
        x_min, x_max, y_min, y_max, z_min, z_max = cuboid
        return abs((x_max - x_min) * (y_max - y_min) * (z_max - z_min))

    def count_cubes_in_initialization_area(self):
        area_x_min = area_y_min = area_z_min = -50
        area_x_max = area_y_max = area_z_max = 51

        area = set()

        for d in self.data:
            on, x_min, x_max, y_min, y_max, z_min, z_max = d
            for x in range(max(x_min, area_x_min), min(x_max + 1, area_x_max)):
                for y in range(max(y_min, area_y_min), min(y_max + 1, area_y_max)):
                    for z in range(max(z_min, area_z_min), min(z_max + 1, area_z_max)):
                        if on:
                            area.add((x, y, z))
                        else:
                            area.discard((x, y, z))

        return len(area)

    def convert_data(self, data):
        ans = []

        for x in data:
            on, x0, x1, y0, y1, z0, z1 = x
            ans.append((on, x0, x1 + 1, y0, y1 + 1, z0, z1 + 1))

        return ans

    def count_all_cubes(self):
        area = set()

        converted_data = self.convert_data(self.data)

        i = 0
        for x in converted_data:
            i += 1
            on = x[0]
            cuboid = x[1:]
            tmp_area = set()
            for c in area:
                new_cuboids = ReactorHandler.cuboid_diff(c, cuboid)

                for nc in new_cuboids:
                    tmp_area.add(nc)

            if on:
                tmp_area.add(cuboid)

            area.clear()
            area = tmp_area.copy()

        ans = 0

        for x in area:
            ans += ReactorHandler.cuboid_volume(x)
        return ans


def parse_day22_data():
    data = []
    with open("day22_a.txt", "r") as f:
        for line in non_blank_lines(f):
            tmp = line.split(" ")
            data.append(tuple([tmp[0] == "on"]) + tuple(map(int, re.findall(r"-?\d+", tmp[1]))))
    return data


def day22_a():
    data = parse_day22_data()
    handler = ReactorHandler(data)
    print("day22_a = {}".format(handler.count_cubes_in_initialization_area()))


def day22_b():
    data = parse_day22_data()
    handler = ReactorHandler(data)
    print("day22_b = {}".format(handler.count_all_cubes()))


class AmphipodHandler:
    def __init__(self, data, large_rooms):
        self.data = data
        self.large_rooms = large_rooms

        self.room_size = 2
        if large_rooms:
            self.room_size *= 2

        self.board_squares = set()
        self.compute_board_squares()

        self.energy_req = {1: 1, 2: 10, 3: 100, 4: 1000}
        self.lowest_energy = sys.maxsize

        self.distances = {}
        self.compute_distances()

        self.amphs = set()
        self.find_amphs()

        self.curr_solution = []
        self.best_solution = []
        self.cache = {}

    def compute_board_squares(self):
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                if self.data[i][j] != 9:
                    self.board_squares.add((i, j))

    class Amph:
        def __init__(self, amph_id, pos):
            self.amph_id = amph_id
            self.pos = pos
            self.target_j = amph_id * 2 + 1

        def move(self, dst):
            self.pos = dst

    def find_amphs(self):
        for s in self.board_squares:
            i, j = s
            if self.data[i][j] != 0:
                amph_id = self.data[i][j]
                self.amphs.add(AmphipodHandler.Amph(amph_id, s))

    def compute_distances(self):
        for j_src in [1, 2, 4, 6, 8, 10, 11]:
            for j_trgt in [3, 5, 7, 9]:
                for i_trgt in range(2, 2 + self.room_size):
                    d = abs(i_trgt - 1) + abs(j_src - j_trgt)
                    self.distances[((1, j_src), (i_trgt, j_trgt))] = d
                    self.distances[((i_trgt, j_trgt), (1, j_src))] = d

        for j_src in [3, 5, 7, 9]:
            for i_src in range(2, 2 + self.room_size):
                for j_trgt in [3, 5, 7, 9]:
                    if j_src == j_trgt:
                        continue
                    for i_trgt in range(2, 2 + self.room_size):
                        d = abs(i_trgt - 1) + abs(i_src - 1) + abs(j_src - j_trgt)
                        self.distances[((i_src, j_src), (i_trgt, j_trgt))] = d

    def game_over(self):
        return all([self.data[i][3:10] == [1, 9, 2, 9, 3, 9, 4] for i in range(2, 2 + self.room_size)])

    def all_possible_moves(self):
        ans = set()
        for amph in self.amphs:
            ans.update({(amph.pos, x) for x in self.possible_moves(amph)})

        return ans

    def possible_moves(self, amph):
        moves = set()
        i, j = amph.pos

        if i == 1:
            cnt = 0
            for a in range(min(j, amph.target_j), max(j, amph.target_j) + 1):
                if self.data[1][a] != 0 and a != j:
                    cnt += 1
            if cnt == 0:
                for target_i in range(2, 2 + self.room_size):
                    if self.data[target_i][amph.target_j] == 0 and all(
                            [self.data[target_i + k][amph.target_j] == amph.amph_id for k in
                             range(1, self.room_size - target_i + 2)]):
                        moves.add((target_i, amph.target_j))
                        break

        if self.data[i - 1][j] == 0 and any([self.data[i + k][j] != (j - 1) // 2 for k in range(self.room_size - i + 2)]):
            for target_j in [1, 2, 4, 6, 8, 10, 11]:
                cnt = 0
                for a in range(min(j, target_j), max(j, target_j) + 1):
                    if self.data[1][a] != 0:
                        cnt += 1
                if cnt == 0:
                    moves.add((1, target_j))

            for target_j in [3, 5, 7, 9]:
                if amph.target_j != target_j:
                    continue
                cnt = 0
                for a in range(min(j, target_j), max(j, target_j) + 1):
                    if self.data[1][a] != 0:
                        cnt += 1
                if cnt == 0:
                    for target_i in range(2, 2 + self.room_size):
                        if self.data[target_i][target_j] == 0 and all([self.data[target_i + k][target_j] == (target_j - 1) // 2 for k in range(1, self.room_size - target_i + 2)]):
                            moves.add((target_i, target_j))
                            break

        return moves

    def move(self, src, trgt):
        src_i, src_j = src
        trgt_i, trgt_j = trgt
        self.data[trgt_i][trgt_j], self.data[src_i][src_j] = self.data[src_i][src_j], self.data[trgt_i][trgt_j]

    def dfs(self, energy, depth):
        if self.game_over():
            return energy

        cache_key = "".join([str(x) for x in self.data])
        if cache_key not in self.cache:
            self.cache[cache_key] = sys.maxsize
            for amph in self.amphs:
                src = amph.pos
                for m in self.possible_moves(amph):
                    dst = m
                    cost = self.energy_req[amph.amph_id] * self.distances[(src, dst)]
                    self.move(src, dst)
                    amph.move(dst)
                    self.curr_solution.append((src, dst))
                    curr = self.dfs(cost, depth + 1)
                    self.cache[cache_key] = min(curr, self.cache[cache_key])
                    self.curr_solution.pop()
                    amph.move(src)
                    self.move(dst, src)

        return energy + self.cache[cache_key]

    def organize_amphipods(self):
        ans = self.dfs(0, 0)
        return ans


def parse_day23a_data():
    mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, '.': 0, ' ': 9, '#': 9}
    with open("day23_a.txt", "r") as f:
        data = [[mapping[x] for x in line] for line in non_blank_lines(f)]

    return data


def day23_a():
    data = parse_day23a_data()
    handler = AmphipodHandler(data, large_rooms=False)
    print("day23_a = {}".format(handler.organize_amphipods()))


def parse_day23b_data():
    mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, '.': 0, ' ': 9, '#': 9}
    with open("day23_a.txt", "r") as f:
        data = [[mapping[x] for x in line] for line in non_blank_lines(f)]

    data = data[0:3] + [[9, 9, 9, 4, 9, 3, 9, 2, 9, 1, 9, 9, 9], [9, 9, 9, 4, 9, 2, 9, 1, 9, 3, 9, 9, 9]] + data[3:]

    return data


def day23_b():
    data = parse_day23b_data()
    handler = AmphipodHandler(data, large_rooms=True)
    print("day23_b = {}".format(handler.organize_amphipods()))


class AssemblyHandler:
    def __init__(self, program):
        self.program = program
        self.regs = [0, 0, 0, 0]
        self.reg_names = {'w': 0, 'x': 1, 'y': 2, 'z': 3}
        self.subprograms = []
        self.parameters = []

    def dump(self):
        w, x, y, z = self.regs
        print("w = {}\nx = {}\ny = {}\nz = {}\n".format(w, x, y, z))

    def dump_short(self):
        print(self.regs)

    def get_regs(self):
        return self.regs

    def clear_regs(self):
        self.regs = [0, 0, 0, 0]

    def split_program(self):
        self.subprograms.clear()
        for x in self.program:
            if x[0] == "inp":
                self.subprograms.append([])

            self.subprograms[-1].append(x)

    def extract_parameters(self):
        self.parameters.clear()
        for x in self.subprograms:
            a, b, c = 0, 0, 0
            a = int(x[4][1].split()[1])
            b = int(x[5][1].split()[1])
            c = int(x[15][1].split()[1])
            self.parameters.append((a, b, c))

    def print_parameters(self):
        print("   n      a    b    c")
        print("----------------------")
        i = 0
        for x in self.parameters:
            a, b, c = x
            print("{:4} | {:4} {:4} {:4}".format(i, a, b, c))
            i += 1
            if i % 5 == 0:
                print()

    def find_largest_model(self):
        self.split_program()
        self.extract_parameters()
        self.print_parameters()
        w = [0] * 14
        stack = []

        for i in range(len(self.parameters)):
            a, b, c = self.parameters[i]
            if a == 1:          # push
                stack.append((i, c))
                w[i] = 9
            elif a == 26:       # pop
                top_i, top_c = stack[-1]
                w[i] = w[top_i] + top_c + b
                if w[i] > 9:
                    w[top_i] -= w[i] - 9
                    w[i] = 9

                stack.pop()

        w_str = "".join(map(str, w))
        if self.run_program(self.program, w_str)[3] == 0:
            return w_str
        else:
            return None

    def find_smallest_model(self):
        self.split_program()
        self.extract_parameters()
        self.print_parameters()
        w = [0] * 14
        stack = []

        for i in range(len(self.parameters)):
            a, b, c = self.parameters[i]
            if a == 1:          # push
                stack.append((i, c))
                w[i] = 1
            elif a == 26:       # pop
                top_i, top_c = stack[-1]
                w[i] = w[top_i] + top_c + b
                if w[i] < 1:
                    w[top_i] = 1 - (top_c + b)
                    w[i] = 1

                stack.pop()

        w_str = "".join(map(str, w))
        if self.run_program(self.program, w_str)[3] == 0:
            return w_str
        else:
            return None

    def run_program(self, program, cin, clear_regs=True):
        if clear_regs:
            self.clear_regs()
        cin = cin[::-1]
        cin = list(cin)
        for line in program:
            instr, args = line
            args = args.split()
            if len(args) == 1:
                args.append(" ")

            a, b = args

            if instr == "inp":
                self.inp(a, cin[-1])
                cin.pop()
            elif instr == "add":
                if b.isdigit():
                    self.add(a, b)
                else:
                    self.add(a, b)
            elif instr == "mul":
                if b.isdigit():
                    self.mul(a, b)
                else:
                    self.mul(a, b)
            elif instr == "div":
                if b.isdigit():
                    self.div(a, b)
                else:
                    self.div(a, b)
            elif instr == "mod":
                if b.isdigit():
                    self.mod(a, b)
                else:
                    self.mod(a, b)
            elif instr == "eql":
                if b.isdigit():
                    self.eql(a, b)
                else:
                    self.eql(a, b)

        # self.dump()
        return self.get_regs()

    @classmethod
    def is_digit(cls, n):
        try:
            int(n)
            return True
        except ValueError:
            return False

    def inp(self, a, b):
        self.regs[self.reg_names[a]] = int(b)

    def add(self, a, b):
        bb = int(b) if AssemblyHandler.is_digit(b) else self.regs[self.reg_names[b]]
        self.regs[self.reg_names[a]] += bb

    def mul(self, a, b):
        bb = int(b) if AssemblyHandler.is_digit(b) else self.regs[self.reg_names[b]]
        self.regs[self.reg_names[a]] *= bb

    def div(self, a, b):
        bb = int(b) if AssemblyHandler.is_digit(b) else self.regs[self.reg_names[b]]
        self.regs[self.reg_names[a]] //= bb

    def mod(self, a, b):
        bb = int(b) if AssemblyHandler.is_digit(b) else self.regs[self.reg_names[b]]
        self.regs[self.reg_names[a]] %= bb

    def eql(self, a, b):
        bb = int(b) if AssemblyHandler.is_digit(b) else self.regs[self.reg_names[b]]
        self.regs[self.reg_names[a]] = 1 if self.regs[self.reg_names[a]] == bb else 0


def parse_day24_data():
    with open("day24_a.txt", "r") as f:
        return [(line.split()[0], " ".join(line.split()[1:])) for line in non_blank_lines(f)]


def day24_a():
    data = parse_day24_data()
    handler = AssemblyHandler(data)
    print("day24_a = {}".format(handler.find_largest_model()))


def day24_b():
    data = parse_day24_data()
    handler = AssemblyHandler(data)
    print("day24_b = {}".format(handler.find_smallest_model()))

