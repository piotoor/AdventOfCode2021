from utilities import non_blank_lines


def draw_horizontal_vertical(diagram, data):
    horizontal_and_vertical = list(filter(lambda x: x[0] == x[2] or x[1] == x[3], data))
    for h in horizontal_and_vertical:
        x0, y0, x1, y1 = h

        if x0 == x1:
            for i in range(min(y0, y1), max(y0, y1) + 1):
                if (x0, i) in diagram:
                    diagram[(x0, i)] += 1
                else:
                    diagram[(x0, i)] = 1
        elif y0 == y1:
            for i in range(min(x0, x1), max(x0, x1) + 1):
                if (i, y0) in diagram:
                    diagram[(i, y0)] += 1
                else:
                    diagram[(i, y0)] = 1

    return diagram


def draw_diagonal(diagram, data):
    diagonal = list(filter(lambda x: abs(x[0] - x[2]) == abs(x[1] - x[3]), data))
    for d in diagonal:
        x0, y0, x1, y1 = d
        curr_x = x0
        curr_y = y0

        if x0 < x1:
            step_x = 1
        else:
            step_x = -1

        if y0 < y1:
            step_y = 1
        else:
            step_y = -1

        for i in range(abs(curr_x - x1) + 1):
            if (curr_x, curr_y) in diagram:
                diagram[(curr_x, curr_y)] += 1
            else:
                diagram[(curr_x, curr_y)] = 1

            curr_x += step_x
            curr_y += step_y

    return diagram


def count_overlapping_horizontal_vertical(data):
    diagram = {}
    diagram = draw_horizontal_vertical(diagram, data)

    vls = diagram.values()
    return len(list(filter(lambda x: x > 1, vls)))


def count_overlapping_horizontal_vertical_diagonal(data):
    diagram = {}
    diagram = draw_horizontal_vertical(diagram, data)
    diagram = draw_diagonal(diagram, data)

    vls = diagram.values()
    return len(list(filter(lambda x: x > 1, vls)))


def parse_day5_data():
    with open("day5.txt", "r") as f:
        data = [line.split(" -> ") for line in non_blank_lines(f)]

    data = [tuple(map(int, (x[0] + "," + x[1]).split(","))) for x in data]
    return data


def day5_a():
    data = parse_day5_data()
    print("day5_a = {}".format(count_overlapping_horizontal_vertical(data)))


def day5_b():
    data = parse_day5_data()
    print("day5_b = {}".format(count_overlapping_horizontal_vertical_diagonal(data)))


def day5():
    day5_a()
    day5_b()
