import sys


def parse_day7_data():
    with open("day7.txt", "r") as f:
        data = list(map(int, f.readline().split(",")))

    return data


def find_optimal_fuel_usage(data, fuel_increase):
    min_x = min(data)
    max_x = max(data)

    optimal_fuel_usage = sys.maxsize

    for i in range(min_x, max_x + 1):
        curr_fuel_usage = 0

        for x in data:
            curr_fuel_usage += fuel_increase(abs(i - x))

        optimal_fuel_usage = min(optimal_fuel_usage, curr_fuel_usage)

    return optimal_fuel_usage


def day7_a():
    data = parse_day7_data()
    print("day7_a = {}".format(find_optimal_fuel_usage(data, lambda x: x)))


def day7_b():
    data = parse_day7_data()
    print("day7_b = {}".format(find_optimal_fuel_usage(data, lambda x: int(0.5 * x * (x + 1)))))


def day7():
    day7_a()
    day7_b()
