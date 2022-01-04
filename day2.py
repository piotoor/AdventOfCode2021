def calculate_hrz_depth_product(data):
    hrz, depth = 0, 0
    for x in data:
        cmd, arg = x

        if cmd == "forward":
            hrz += int(arg)
        elif cmd == "up":
            depth -= int(arg)
        elif cmd == "down":
            depth += int(arg)

    return hrz * depth


def calculate_hrz_depth_aim_product(data):
    hrz, depth, aim = 0, 0, 0
    for x in data:
        cmd, arg = x

        if cmd == "forward":
            hrz += int(arg)
            depth += aim * int(arg)
        elif cmd == "up":
            aim -= int(arg)
        elif cmd == "down":
            aim += int(arg)

    return hrz * depth


def parse_day2_data():
    with open("day2.txt", "r") as f:
        data = [tuple(cmd.split()) for cmd in f.read().splitlines()]

    return data


def day2_a():
    data = parse_day2_data()
    print("day2_a = {}".format(calculate_hrz_depth_product(data)))


def day2_b():
    data = parse_day2_data()
    print("day2_b = {}".format(calculate_hrz_depth_aim_product(data)))
