from abc import ABC, abstractmethod
from utilities import non_blank_lines


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


def parse_day4_data():
    raw_data = []
    with open("day4.txt", "r") as f:
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


def day4():
    day4_a()
    day4_b()
