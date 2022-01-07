import collections
import sys
import copy
import math
from operator import add
import re
from utilities import non_blank_lines
from math import copysign

sys.setrecursionlimit(15000)



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
    with open("day18.txt", "r") as f:
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
    with open("day19.txt", "r") as f:
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
    with open("day20.txt", "r") as f:
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
    with open("day21.txt", "r") as f:
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
    with open("day22.txt", "r") as f:
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
    with open("day23.txt", "r") as f:
        data = [[mapping[x] for x in line] for line in non_blank_lines(f)]

    return data


def day23_a():
    data = parse_day23a_data()
    handler = AmphipodHandler(data, large_rooms=False)
    print("day23_a = {}".format(handler.organize_amphipods()))


def parse_day23b_data():
    mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, '.': 0, ' ': 9, '#': 9}
    with open("day23.txt", "r") as f:
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
    with open("day24.txt", "r") as f:
        return [(line.split()[0], " ".join(line.split()[1:])) for line in non_blank_lines(f)]


def day24_a():
    data = parse_day24_data()
    handler = AssemblyHandler(data)
    print("day24_a = {}".format(handler.find_largest_model()))


def day24_b():
    data = parse_day24_data()
    handler = AssemblyHandler(data)
    print("day24_b = {}".format(handler.find_smallest_model()))


class CucumberHandler:
    def __init__(self, data):
        self.board = data

    def count_steps(self):
        moved = True
        step_count = 0
        rr = len(self.board)
        cc = len(self.board[0])

        while moved:
            step_count += 1
            moved = False
            r = 0
            while r < rr:
                c = 0
                first = self.board[r][0]
                last = self.board[r][-1]
                while c < cc:
                    if self.board[r][c] == '>':
                        if self.board[r][(c + 1) % cc] == '.':
                            # print("{} -> {}".format((r, c), (r, (c + 1) % cc)))
                            self.board[r][c], self.board[r][(c + 1) % cc] = self.board[r][(c + 1) % cc], self.board[r][c]
                            moved = True
                            c += 1
                    c += 1
                if first == '>' and last == '>' and self.board[r][-1] == '.':
                    self.board[r][0], self.board[r][-1] = self.board[r][-1], self.board[r][0]
                r += 1

            c = 0
            while c < cc:
                r = 0
                first = self.board[0][c]
                last = self.board[-1][c]
                while r < rr:
                    if self.board[r][c] == 'v':
                        if self.board[(r + 1) % rr][c] == '.':
                            self.board[r][c], self.board[(r + 1) % rr][c] = self.board[(r + 1) % rr][c], self.board[r][c]
                            moved = True
                            r += 1
                    r += 1
                if first == 'v' and last == 'v' and self.board[-1][c] == '.':
                    self.board[0][c], self.board[-1][c] = self.board[-1][c], self.board[0][c]
                c += 1

        return step_count


def parse_day25_data():
    with open("day25.txt", "r") as f:
        data = [list(line) for line in non_blank_lines(f)]
    return data


def day25_a():
    data = parse_day25_data()
    handler = CucumberHandler(data)
    print("day25_a = {}".format(handler.count_steps()))
