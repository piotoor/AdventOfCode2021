class ProbeLauncher:
    def __init__(self, data):
        self.target_area = data
        self.highest_y_values = []
        self.initial_velocities = set()

    def cleanup(self):
        self.highest_y_values.clear()
        self.initial_velocities.clear()

    def compute(self):
        self.cleanup()
        t_x0, t_x1, t_y0, t_y1 = self.target_area
        start_x, start_y = 0, 0
        max_v_y = 150

        for vx_init in range(0, t_x1 + 1):
            for vy_init in range(-max_v_y, max_v_y):
                v_x = vx_init
                v_y = vy_init
                pos_x = start_x
                pos_y = start_y
                max_y = pos_y

                while pos_x <= t_x1 and pos_y >= t_y0:
                    pos_x += v_x
                    pos_y += v_y
                    max_y = max(max_y, pos_y)

                    if t_x0 <= pos_x <= t_x1 and t_y0 <= pos_y <= t_y1:
                        self.highest_y_values.append(max_y)
                        self.initial_velocities.add((vx_init, vy_init))
                        break

                    if v_x > 0:
                        v_x -= 1
                    v_y -= 1

    def find_highest_y(self):
        return max(self.highest_y_values)

    def count_initial_velocities(self):
        return len(self.initial_velocities)


def parse_day17_data():
    data = []
    with open("day17.txt", "r") as f:
        line = f.readline()[13:].split(", ")
        for coords in line:
            data.extend((coords[2:].split("..")))

    return tuple(int(x) for x in data)


def day17_a():
    data = parse_day17_data()
    launcher = ProbeLauncher(data)
    launcher.compute()
    print("day17_a = {}".format(launcher.find_highest_y()))


def day17_b():
    data = parse_day17_data()
    launcher = ProbeLauncher(data)
    launcher.compute()
    print("day17_b = {}".format(launcher.count_initial_velocities()))


def day17():
    day17_a()
    day17_b()
