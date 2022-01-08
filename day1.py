def count_number_of_increases(data):
    ans = 0

    for i in range(1, len(data)):
        if data[i] > data[i - 1]:
            ans += 1

    return ans


def count_number_of_increased_windows(data):
    ans = 0

    if len(data) <= 3:
        return ans

    prev = sum(data[0:3])
    for i in range(3, len(data) + 1):
        curr = sum(data[i - 3:i])
        if curr > prev:
            ans += 1
        prev = curr

    return ans


def parse_day1_data():
    with open("day1.txt", "r") as f:
        data = list(map(int, f.read().splitlines()))

    return data


def day1_a():
    data = parse_day1_data()
    print("day1_a = {}".format(count_number_of_increases(data)))


def day1_b():
    data = parse_day1_data()
    print("day1_b = {}".format(count_number_of_increased_windows(data)))


def day1():
    day1_a()
    day1_b()
