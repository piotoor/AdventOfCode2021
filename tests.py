import solutions
import unittest
from parameterized import parameterized


class Day1(unittest.TestCase):
    def test_count_number_of_increases_empty(self):
        data = []
        expected = 0
        self.assertEqual(expected, solutions.count_number_of_increases(data))

    def test_count_number_of_increases(self):
        data = [199, 200, 208, 210, 200, 207, 240, 269, 260, 263]
        expected = 7
        self.assertEqual(expected, solutions.count_number_of_increases(data))

    def test_count_number_of_increased_windows_empty(self):
        data = []
        expected = 0
        self.assertEqual(expected, solutions.count_number_of_increased_windows(data))

    def test_count_number_of_increased_windows_up_to_3_elements(self):
        data = [1]
        expected = 0
        self.assertEqual(expected, solutions.count_number_of_increased_windows(data))

        data = [1, 10]
        expected = 0
        self.assertEqual(expected, solutions.count_number_of_increased_windows(data))

        data = [1, 10, 100]
        expected = 0
        self.assertEqual(expected, solutions.count_number_of_increased_windows(data))

    def test_count_number_of_increased_windows(self):
        data = [199, 200, 208, 210, 200, 207, 240, 269, 260, 263]
        expected = 5
        self.assertEqual(expected, solutions.count_number_of_increased_windows(data))


class Day2(unittest.TestCase):
    def test_calculate_hrz_depth_product(self):
        data = [("forward", "5"),
                ("down", "5"),
                ("forward", "8"),
                ("up", "3"),
                ("down", "8"),
                ("forward", "2")]
        expected = 150
        self.assertEqual(expected, solutions.calculate_hrz_depth_product(data))

    def test_calculate_hrz_depth_aim_product(self):
        data = [("forward", "5"),
                ("down", "5"),
                ("forward", "8"),
                ("up", "3"),
                ("down", "8"),
                ("forward", "2")]
        expected = 900
        self.assertEqual(expected, solutions.calculate_hrz_depth_aim_product(data))


class Day3(unittest.TestCase):
    def setUp(self):
        self.data = ['00100',
                     '11110',
                     '10110',
                     '10111',
                     '10101',
                     '01111',
                     '00111',
                     '11100',
                     '10000',
                     '11001',
                     '00010',
                     '01010']

    def test_calculate_power_consumption(self):
        expected = 198
        self.assertEqual(expected, solutions.calculate_power_consumption(self.data))

    def test_calculate_oxygen_generator_rating(self):
        expected = 23
        self.assertEqual(expected, solutions.calculate_oxygen_generator_rating(self.data))

    def test_calculate_co2_scrubber_rating(self):
        expected = 10
        self.assertEqual(expected, solutions.calculate_co2_scrubber_rating(self.data))

    def test_calculate_life_support_rating(self):
        expected = 230
        self.assertEqual(expected, solutions.calculate_life_support_rating(self.data))


class Day4(unittest.TestCase):
    def setUp(self):
        self.boards = [
            [[22, 13, 17, 11, 0],
             [8, 2, 23, 4, 24],
             [21, 9, 14, 16, 7],
             [6, 10, 3, 18, 5],
             [1, 12, 20, 15, 19]],

            [[3, 15, 0, 2, 22],
             [9, 18, 13, 17, 5],
             [19, 8, 7, 25, 23],
             [20, 11, 10, 24, 4],
             [14, 21, 16, 12, 6]],

            [[14, 21, 17, 24, 4],
             [10, 16, 15, 9, 19],
             [18, 8, 23, 26, 20],
             [22, 11, 13, 6, 5],
             [2, 0, 12, 3, 7]]
        ]

        self.nums = [7, 4, 9, 5, 11, 17, 23, 2, 0, 14, 21, 24, 10, 16, 13, 6, 15, 25, 12, 22, 18, 20, 8, 19, 3, 26, 1]

    def test_get_first_winner_points(self):
        bingo = solutions.Bingo(self.boards, self.nums)
        expected_winner_board_points = 4512

        while bingo.is_move_possible():
            bingo.step()

        self.assertEqual(expected_winner_board_points, bingo.get_winner_points())

    def test_get_last_winner_points(self):
        bingo = solutions.AntiBingo(self.boards, self.nums)
        expected_winner_board_points = 1924

        while bingo.is_move_possible():
            bingo.step()

        self.assertEqual(expected_winner_board_points, bingo.get_winner_points())


class Day5(unittest.TestCase):
    def setUp(self):
        self.data = [
            (0, 9, 5, 9),
            (8, 0, 0, 8),
            (9, 4, 3, 4),
            (2, 2, 2, 1),
            (7, 0, 7, 4),
            (6, 4, 2, 0),
            (0, 9, 2, 9),
            (3, 4, 1, 4),
            (0, 0, 8, 8),
            (5, 5, 8, 2)
        ]

    def test_count_overlapping_horizontal_vertical(self):
        expected = 5
        self.assertEqual(expected, solutions.count_overlapping_horizontal_vertical(self.data))

    def test_count_overlapping_horizontal_vertical_diagonal(self):
        expected = 12
        self.assertEqual(expected, solutions.count_overlapping_horizontal_vertical_diagonal(self.data))


class Day6(unittest.TestCase):
    def setUp(self):
        self.data = [3, 4, 3, 1, 2]
        self.days = 80
        self.expected = 5934

    def test_count_lanternfish(self):
        self.assertEqual(self.expected, solutions.count_lanternfish(self.data, self.days))


class Day7(unittest.TestCase):
    @parameterized.expand([
        ("const fuel usage", [16, 1, 2, 0, 4, 2, 7, 1, 2, 14], 37, lambda x: x),
        ("increasing fuel usage", [16, 1, 2, 0, 4, 2, 7, 1, 2, 14], 168, lambda x: int(0.5 * x * (x + 1))),
    ])
    def test_find_optimal_fuel_usage(self, name, data, expected, increase):
        self.assertEqual(expected, solutions.find_optimal_fuel_usage(data, increase))


class Day8(unittest.TestCase):
    def setUp(self):
        self.raw_data = [
            "be cfbegad cbdgef fgaecd cgeb fdcge agebfd fecdb fabcd edb | fdgacbe cefdb cefbgd gcbe",
            "edbfga begcd cbg gc gcadebf fbgde acbgfd abcde gfcbed gfec | fcgedb cgb dgebacf gc",
            "fgaebd cg bdaec gdafb agbcfd gdcbef bgcad gfac gcb cdgabef | cg cg fdcagb cbg",
            "fbegcd cbd adcefb dageb afcb bc aefdc ecdab fgdeca fcdbega | efabcd cedba gadfec cb",
            "aecbfdg fbg gf bafeg dbefa fcge gcbea fcaegb dgceab fcbdga | gecf egdcabf bgf bfgea",
            "fgeab ca afcebg bdacfeg cfaedg gcfdb baec bfadeg bafgc acf | gebdcfa ecba ca fadegcb",
            "dbcfg fgd bdegcaf fgec aegbdf ecdfab fbedc dacgb gdcebf gf | cefg dcbef fcge gbcadfe",
            "bdfegc cbegaf gecbf dfcage bdacg ed bedf ced adcbefg gebcd | ed bcgafe cdgba cbgef",
            "egadfb cdbfeg cegd fecab cgb gbdefca cg fgcdab egfdb bfceg | gbdfcae bgc cg cgb",
            "gcafb gcf dcaebfg ecagb gf abcdeg gaef cafbge fdbac fegbdc | fgae cfgab fg bagce"
        ]

    def test_count_digits_made_of_unique_number_of_segments(self):
        expected = 26
        data = [[x.split() for x in line.split(" | ")] for line in self.raw_data]
        self.assertEqual(expected, solutions.count_digits_made_of_unique_number_of_segments(data))

    def test_calculate_sum_of_all_output_digits(self):
        expected = 61229
        data = [[x.split() for x in line.split(" | ")] for line in self.raw_data]
        self.assertEqual(expected, solutions.calculate_sum_of_all_output_digits(data))
