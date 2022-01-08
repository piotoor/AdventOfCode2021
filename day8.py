from utilities import non_blank_lines


def count_digits_made_of_unique_number_of_segments(data):
    number_of_digits_of_unique_seqments = 0

    for x in data:
        output = x[1]
        # print("output = {}, filtered = {}".format(output, list(filter(lambda y: len(y) in [2, 3, 4, 7], output))))
        number_of_digits_of_unique_seqments += len(list(filter(lambda y: len(y) in [2, 3, 4, 7], output)))

    return number_of_digits_of_unique_seqments


def calculate_sum_of_all_output_digits(data):
    ans = 0

    for line in data:
        signal_patterns = line[0]
        identified_digits = [set()] * 10
        remaining_digits = []
        segments = [set()] * 7

        for digit in signal_patterns:
            if len(digit) == 2:
                identified_digits[1] = set(digit)
            elif len(digit) == 4:
                identified_digits[4] = set(digit)
            elif len(digit) == 3:
                identified_digits[7] = set(digit)
            elif len(digit) == 7:
                identified_digits[8] = set(digit)
            else:
                remaining_digits.append(set(digit))

        segments[0] = identified_digits[7] - identified_digits[1]

        for x in remaining_digits:  # 9, 6, 0
            if len(x) == 6 and len(x - identified_digits[7] - identified_digits[4]) == 1:
                segments[6] = x - identified_digits[7] - identified_digits[4]
                break

        segments[4] = identified_digits[8] - segments[0] - segments[6] - identified_digits[4]

        for x in remaining_digits:  # 5, 3, 2
            if len(x) == 5 and len(x - segments[0] - segments[4] - segments[6] - identified_digits[1]) == 1:
                segments[3] = x - segments[0] - segments[4] - segments[6] - identified_digits[1]
                break

        segments[1] = identified_digits[4] - segments[3] - identified_digits[1]

        for x in remaining_digits:  # 9, 6, 0
            if len(x) == 6 and len(x - segments[0] - segments[1] - segments[3] - segments[4] - segments[6]) == 1:
                segments[5] = x - segments[0] - segments[1] - segments[3] - segments[4] - segments[6]
                break

        all_segments = set()
        for x in segments:
            if x is not None:
                all_segments = all_segments | x

        segments[2] = identified_digits[8] - all_segments

        identified_digits[0] = identified_digits[8] - segments[3]
        identified_digits[2] = identified_digits[8] - segments[1] - segments[5]
        identified_digits[3] = identified_digits[8] - segments[1] - segments[4]
        identified_digits[5] = identified_digits[8] - segments[2] - segments[4]
        identified_digits[6] = identified_digits[8] - segments[2]
        identified_digits[9] = identified_digits[8] - segments[4]

        identified_digits_dict = {}
        for i in range(len(identified_digits)):
            identified_digits_dict["".join(sorted(identified_digits[i]))] = i

        output_num = ""
        for output_digs in line[1]:
            output_num = output_num + str(identified_digits_dict["".join(sorted(output_digs))])

        ans += int(output_num)

    return ans


def parse_day8_data():
    with open("day8.txt", "r") as f:
        data = [[x.split() for x in line.split(" | ")] for line in non_blank_lines(f)]

    return data


def day8_a():
    data = parse_day8_data()
    print("day8_a = {}".format(count_digits_made_of_unique_number_of_segments(data)))


def day8_b():
    data = parse_day8_data()
    print("day8_a = {}".format(calculate_sum_of_all_output_digits(data)))


def day8():
    day8_a()
    day8_b()
