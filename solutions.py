import collections
import sys
import copy
import math
from operator import add
import re
from utilities import non_blank_lines
from math import copysign

sys.setrecursionlimit(15000)


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
