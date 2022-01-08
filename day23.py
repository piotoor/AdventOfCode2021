import sys
from utilities import non_blank_lines


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


def parse_day23b_data():
    mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, '.': 0, ' ': 9, '#': 9}
    with open("day23.txt", "r") as f:
        data = [[mapping[x] for x in line] for line in non_blank_lines(f)]

    data = data[0:3] + [[9, 9, 9, 4, 9, 3, 9, 2, 9, 1, 9, 9, 9], [9, 9, 9, 4, 9, 2, 9, 1, 9, 3, 9, 9, 9]] + data[3:]

    return data


def day23_a():
    data = parse_day23a_data()
    handler = AmphipodHandler(data, large_rooms=False)
    print("day23_a = {}".format(handler.organize_amphipods()))


def day23_b():
    data = parse_day23b_data()
    handler = AmphipodHandler(data, large_rooms=True)
    print("day23_b = {}".format(handler.organize_amphipods()))
