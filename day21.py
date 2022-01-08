import copy
from operator import add
from utilities import non_blank_lines


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


def day21():
    day21_a()
    day21_b()
