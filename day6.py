def count_lanternfish(lanternfish, days):
    lanternfish_map = {}
    for x in lanternfish:
        if x in lanternfish_map:
            lanternfish_map[x] += 1
        else:
            lanternfish_map[x] = 1

    while days > 0:
        tmp_map = {}
        for k, v in lanternfish_map.items():
            if k - 1 < 0:
                if 8 in tmp_map:
                    tmp_map[8] += v
                else:
                    tmp_map[8] = v

                if 6 in tmp_map:
                    tmp_map[6] += v
                else:
                    tmp_map[6] = v
            else:
                if k - 1 in tmp_map:
                    tmp_map[k - 1] += v
                else:
                    tmp_map[k - 1] = v

        lanternfish_map = tmp_map
        days -= 1

    return sum(lanternfish_map.values())


def parse_day6_data():
    with open("day6.txt", "r") as f:
        data = list(map(int, f.readline().split(",")))

    return data


def day6_a():
    data = parse_day6_data()
    print("day6_a = {}".format(count_lanternfish(data, 80)))


def day6_b():
    data = parse_day6_data()
    print("day6_b = {}".format(count_lanternfish(data, 256)))


def day6():
    day6_a()
    day6_b()
