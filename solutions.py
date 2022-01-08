import collections
import sys
import copy
import math


from utilities import non_blank_lines
from math import copysign

sys.setrecursionlimit(15000)




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
