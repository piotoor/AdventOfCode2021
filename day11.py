from utilities import non_blank_lines


def parse_day11_data():
    with open("day11.txt", "r") as f:
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
