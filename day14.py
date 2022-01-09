from utilities import non_blank_lines
from collections import Counter


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


def parse_day14_data():
    data = []

    with open("day14.txt", "r") as f:
        data.append(f.readline().rstrip())
        for line in non_blank_lines(f):
            data.append(tuple(line.split(" -> ")))

    return data


def day14_a():
    data = parse_day14_data()
    poly_handler = PolymerHandler(data)
    print("day14_a = {}".format(poly_handler.most_common_least_common_diff(10)))


def day14_b():
    data = parse_day14_data()
    poly_handler = PolymerHandler(data)
    print("day14_b = {}".format(poly_handler.most_common_least_common_diff(40)))


def day14():
    day14_a()
    day14_b()
