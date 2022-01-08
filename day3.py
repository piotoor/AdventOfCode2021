def calculate_power_consumption(data):
    bit_count = [0] * len(data[0])
    for x in data:
        for i in range(len(x)):
            bit_count[i] += int(x[i])

    gamma_rate_list = [1 if x >= len(data) / 2 else 0 for x in bit_count]
    epsilon_rate_list = [1 if x < len(data) / 2 else 0 for x in bit_count]

    gamma_rate = int("".join(map(str, gamma_rate_list)), 2)
    epsilon_rate = int("".join(map(str, epsilon_rate_list)), 2)
    return gamma_rate * epsilon_rate


def calculate_rating(data, bit):
    curr_data = data.copy()

    i = 0
    while len(curr_data) > 1 and i < len(data[0]):
        bit_count_ith_column = sum(bit if x[i] == str(bit) else int(not bit) for x in curr_data)

        if bit_count_ith_column >= len(curr_data) / 2:
            curr_data = list(filter(lambda a: a[i] == str(bit), curr_data))
        else:
            curr_data = list(filter(lambda a: a[i] == str(int(not bool(bit))), curr_data))

        i += 1

    return int(curr_data[0], 2)


def calculate_oxygen_generator_rating(data):
    return calculate_rating(data, 1)


def calculate_co2_scrubber_rating(data):
    return calculate_rating(data, 0)


def calculate_life_support_rating(data):
    return calculate_oxygen_generator_rating(data) * calculate_co2_scrubber_rating(data)


def parse_day3_data():
    with open("day3.txt", "r") as f:
        data = list(f.read().splitlines())

    return data


def day3_a():
    data = parse_day3_data()
    print("day3_a = {}".format(calculate_power_consumption(data)))


def day3_b():
    data = parse_day3_data()
    print("day3_b = {}".format(calculate_life_support_rating(data)))


def day3():
    day3_a()
    day3_b()
