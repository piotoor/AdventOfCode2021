import operator
from abc import ABC, abstractmethod
import sys
import itertools
import statistics

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
        if r < 0 or c < 0 or r >= len(self.visited) or c >= len(self.visited[0]) or self.visited[r][c] in [9, self.curr]:
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
