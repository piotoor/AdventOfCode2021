from utilities import non_blank_lines
import statistics


class ParenthesisParser:
    brackets_pairs = {'[': ']', '(': ')', '<': '>', '{': '}'}
    opening_brackets = {'[', '(', '<', '{'}
    syntax_error_points = {
        ')': 3,
        ']': 57,
        '}': 1197,
        '>': 25137,
    }
    autocomplete_points = {
        ')': 1,
        ']': 2,
        '}': 3,
        '>': 4,
    }

    def __init__(self, data):
        self.data = data
        self.invalid_brackets = []
        self.missing_brackets = []

    @classmethod
    def find_first_corrupted_bracket_in_expr(cls, expr):
        stack = []

        for x in expr:
            if x in ParenthesisParser.opening_brackets:
                stack.append(x)
            elif ParenthesisParser.brackets_pairs[stack[-1]] == x:
                stack.pop()
            else:
                return x

        return None

    def find_invalid_brackets(self):
        for row in self.data:
            first_corrupted = ParenthesisParser.find_first_corrupted_bracket_in_expr(row)
            if first_corrupted is not None:
                self.invalid_brackets.append(first_corrupted)

    def calculate_syntax_error_score(self):
        self.find_invalid_brackets()
        ans = 0

        for x in self.invalid_brackets:
            ans += ParenthesisParser.syntax_error_points[x]

        return ans

    @classmethod
    def find_missing_brackets_in_expr(cls, expr):
        stack = []

        for x in expr:
            if x in ParenthesisParser.opening_brackets:
                stack.append(x)
            elif ParenthesisParser.brackets_pairs[stack[-1]] == x:
                stack.pop()
            else:
                return None

        return "".join(reversed(list(map(lambda y: ParenthesisParser.brackets_pairs[y], stack))))

    def find_missing_brackets(self):
        for row in self.data:
            missing_brackets = ParenthesisParser.find_missing_brackets_in_expr(row)
            if missing_brackets is not None:
                self.missing_brackets.append(missing_brackets)

    def calculate_autocomplete_score(self):
        self.find_missing_brackets()
        scores = [0] * len(self.missing_brackets)

        for i in range(len(self.missing_brackets)):
            for x in self.missing_brackets[i]:
                scores[i] *= 5
                scores[i] += ParenthesisParser.autocomplete_points[x]

        return statistics.median(scores)


def parse_day10_data():
    with open("day10.txt", "r") as f:
        data = [line for line in non_blank_lines(f)]

    return data


def day10_a():
    data = parse_day10_data()
    parser = ParenthesisParser(data)

    print("day10_a = {}".format(parser.calculate_syntax_error_score()))


def day10_b():
    data = parse_day10_data()
    parser = ParenthesisParser(data)

    print("day10_b = {}".format(parser.calculate_autocomplete_score()))
