import copy
from utilities import non_blank_lines
import math


class SnailfishCalculator:
    def __init__(self, data):
        self.data = data

    @classmethod
    def explode(cls, number):
        new_number = copy.deepcopy(number)
        to_explode_indices = []
        to_update_indices = [-1, -1]

        for i in range(len(new_number)):
            if len(to_explode_indices) < 2:
                if new_number[i][1] == 4:
                    to_explode_indices.append(i)
                else:
                    to_update_indices[0] = i
            elif to_update_indices[1] == -1:
                to_update_indices[1] = i

        if len(to_explode_indices) < 2:
            return False, new_number
        if to_update_indices[0] != -1:
            new_number[to_update_indices[0]][0] += new_number[to_explode_indices[0]][0]

        if to_update_indices[1] != -1:
            new_number[to_update_indices[1]][0] += new_number[to_explode_indices[1]][0]

        new_number[to_explode_indices[0]] = [0, 3]
        del new_number[to_explode_indices[1]]

        return True, new_number

    @classmethod
    def split(cls, number):
        new_number = copy.deepcopy(number)

        for i in range(len(new_number)):
            curr_val = new_number[i][0]
            curr_depth = new_number[i][1]

            if curr_val > 9:
                left = int(math.floor(curr_val / 2.0))
                right = int(math.ceil(curr_val / 2.0))
                new_number[i] = [left, curr_depth + 1]
                new_number.insert(i + 1, [right, curr_depth + 1])
                return True, new_number

        return False, new_number

    @classmethod
    def increase_depth(cls, number):
        new_number = copy.deepcopy(number)

        for i in range(len(new_number)):
            new_number[i][1] += 1

        return new_number

    @classmethod
    def add(cls, n1, n2):
        ans = copy.deepcopy(n1)
        ans.extend(n2)
        ans = SnailfishCalculator.increase_depth(ans)

        return ans

    @classmethod
    def chain_add(cls, numbers):
        ans = None

        for x in numbers:
            if ans is None:
                ans = x
                continue

            ans = SnailfishCalculator.add(ans, x)
            reduce = True
            splitted = False
            while reduce:
                exploded, ans = SnailfishCalculator.explode(ans)
                if not exploded:
                    splitted, ans = SnailfishCalculator.split(ans)

                reduce = exploded or splitted

        return ans

    @classmethod
    def calculate_magnitude(cls, number):
        stack = []

        for x in number:
            curr = x

            while len(stack) > 0 and stack[-1][1] == curr[1]:
                curr = [3 * stack[-1][0] + 2 * curr[0], curr[1] - 1]
                stack.pop()

            stack.append(curr)

        return stack[-1][0]

    def calculate_total_magnitude(self):
        final_number = SnailfishCalculator.chain_add(self.data)
        return SnailfishCalculator.calculate_magnitude(final_number)

    def calculate_largest_sum_of_two_numbers(self):
        max_sum = 0
        for a in self.data:
            for b in self.data:
                if a != b:
                    a_b = SnailfishCalculator.chain_add([a, b])
                    b_a = SnailfishCalculator.chain_add([b, a])
                    a_b_magnitude = SnailfishCalculator.calculate_magnitude(a_b)
                    b_a_magnitude = SnailfishCalculator.calculate_magnitude(b_a)
                    max_sum = max(max_sum, a_b_magnitude, b_a_magnitude)

        return max_sum


def parse_day18_data():
    with open("day18.txt", "r") as f:
        raw_data = [line for line in non_blank_lines(f)]

    ans = []

    for line in raw_data:
        converted_line = []
        depth = -1
        for x in line:
            if x == '[':
                depth += 1
            elif x == ']':
                depth -= 1
            elif x.isdigit():
                converted_line.append([int(x), depth])
        ans.append(converted_line)

    return ans


def day18_a():
    data = parse_day18_data()
    calc = SnailfishCalculator(data)
    print("day18_a = {}".format(calc.calculate_total_magnitude()))


def day18_b():
    data = parse_day18_data()
    calc = SnailfishCalculator(data)
    print("day18_b = {}".format(calc.calculate_largest_sum_of_two_numbers()))


def day18():
    day18_a()
    day18_b()
