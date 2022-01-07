from utilities import non_blank_lines


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


def parse_day13_data():
    dots = set()
    instructions = []
    with open("day13.txt", "r") as f:
        for line in non_blank_lines(f):
            if line[0] == 'f':
                fold_axis = line.split(" ")[2].split("=")
                instructions.append((fold_axis[0], int(fold_axis[1])))

            else:
                dots.add(tuple(map(int, line.split(","))))

    return dots, instructions


def day13_a():
    data = parse_day13_data()
    folder = OrigamiFolder(data)
    print("day13_a = {}".format(folder.count_dots_after_single_fold()))


def day13_b():
    data = parse_day13_data()
    folder = OrigamiFolder(data)
    print("day13_a = {}".format(folder.count_dots_after_full_fold()))
    folder.print()
