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
    def test_find_optimal_fuel_usage(self, _, data, expected, increase):
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

        self.data = [[x.split() for x in line.split(" | ")] for line in self.raw_data]

    def test_count_digits_made_of_unique_number_of_segments(self):
        expected = 26
        self.assertEqual(expected, solutions.count_digits_made_of_unique_number_of_segments(self.data))

    def test_calculate_sum_of_all_output_digits(self):
        expected = 61229
        self.assertEqual(expected, solutions.calculate_sum_of_all_output_digits(self.data))


class Day9(unittest.TestCase):
    @parameterized.expand([
        ("aoc example", [
            [2, 1, 9, 9, 9, 4, 3, 2, 1, 0],
            [3, 9, 8, 7, 8, 9, 4, 9, 2, 1],
            [9, 8, 5, 6, 7, 8, 9, 8, 9, 2],
            [8, 7, 6, 7, 8, 9, 6, 7, 8, 9],
            [9, 8, 9, 9, 9, 6, 5, 6, 7, 8]
        ], 15),
        ("own example 1", [
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        ], 4),
        ("own example 2", [
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 3, 3, 3, 1, 1, 1, 1],
            [1, 1, 1, 3, 1, 3, 1, 1, 1, 1],
            [1, 1, 1, 3, 3, 3, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        ], 6),

    ])
    def test_calculate_sum_of_the_risk_levels(self, _, data, expected):
        self.assertEqual(expected, solutions.calculate_sum_of_the_risk_levels(data))

    @parameterized.expand([
        ("aoc example", [
            [2, 1, 9, 9, 9, 4, 3, 2, 1, 0],
            [3, 9, 8, 7, 8, 9, 4, 9, 2, 1],
            [9, 8, 5, 6, 7, 8, 9, 8, 9, 2],
            [8, 7, 6, 7, 8, 9, 6, 7, 8, 9],
            [9, 8, 9, 9, 9, 6, 5, 6, 7, 8]
        ], 1134),
    ])
    def test_calculate_sum_of_three_largest_basins(self, _, data, expected):
        basin_handler = solutions.BasinHandler(data)
        self.assertEqual(expected, basin_handler.calculate_sum_of_three_largest_basins())


class Day10(unittest.TestCase):
    @parameterized.expand([
        ("aoc example 1", "[({(<(())[]>[[{[]{<()<>>", None),
        ("aoc example 2", "[(()[<>])]({[<{<<[]>>(", None),
        ("aoc example 3", "{([(<{}[<>[]}>{[]{[(<()>", '}'),
        ("aoc example 4", "(((({<>}<{<{<>}{[]{[]{}", None),
        ("aoc example 5", "[[<[([]))<([[{}[[()]]]", ')'),
        ("aoc example 6", "[{[{({}]{}}([{[{{{}}([]", ']'),
        ("aoc example 7", "{<[[]]>}<{[{[{[]{()[[[]", None),
        ("aoc example 8", "[<(<(<(<{}))><([]([]()", ')'),
        ("aoc example 9", "<{([([[(<>()){}]>(<<{{", '>'),
        ("aoc example 10", "<{([{{}}[<[[[<>{}]]]>[]]", None),
        ("own example 1", "<<<<>>>", None),
        ("own example 2", "<<<<>>>)", ')')
    ])
    def test_find_first_corrupted_bracket_in_expr(self, _, expression, expected):
        self.assertEqual(expected, solutions.ParenthesisParser.find_first_corrupted_bracket_in_expr(expression))

    @parameterized.expand([
        ("own example 1", [
            "<<<<>>>)"
        ], 3),
        ("own example 2", [
            "<<<<>>>]"
        ], 57),
        ("own example 3", [
            "<<<<>>>}"
        ], 1197),
        ("own example 4", [
            "(>"
        ], 25137),
        ("own example 5", [
            "<<<<>>>)",
            "<<<<>>>)",
            "<<<<>>>]",
            "<<<<>>>]",
            "<<<<>>>}",
            "<<<<>>>}",
            "(>",
            "(>"
        ], 52788),
        ("aoc example", [
            "[({(<(())[]>[[{[]{<()<>>",
            "[(()[<>])]({[<{<<[]>>(",
            "{([(<{}[<>[]}>{[]{[(<()>",
            "(((({<>}<{<{<>}{[]{[]{}",
            "[[<[([]))<([[{}[[()]]]",
            "[{[{({}]{}}([{[{{{}}([]",
            "{<[[]]>}<{[{[{[]{()[[[]",
            "[<(<(<(<{}))><([]([]()",
            "<{([([[(<>()){}]>(<<{{"
            "<{([{{}}[<[[[<>{}]]]>[]]"
        ], 26397),
    ])
    def test_calculate_syntax_error_score(self, _, data, expected):
        parser = solutions.ParenthesisParser(data)
        self.assertEqual(expected, parser.calculate_syntax_error_score())

    @parameterized.expand([
        ("aoc example 1", "[({(<(())[]>[[{[]{<()<>>", "}}]])})]"),
        ("aoc example 2", "[(()[<>])]({[<{<<[]>>(", ")}>]})"),
        ("aoc example 3", "{([(<{}[<>[]}>{[]{[(<()>", None),
        ("aoc example 4", "(((({<>}<{<{<>}{[]{[]{}", "}}>}>))))"),
        ("aoc example 5", "[[<[([]))<([[{}[[()]]]", None),
        ("aoc example 6", "[{[{({}]{}}([{[{{{}}([]", None),
        ("aoc example 7", "{<[[]]>}<{[{[{[]{()[[[]", "]]}}]}]}>"),
        ("aoc example 8", "[<(<(<(<{}))><([]([]()", None),
        ("aoc example 9", "<{([([[(<>()){}]>(<<{{", None),
        ("aoc example 10", "<{([{{}}[<[[[<>{}]]]>[]]", "])}>"),
    ])
    def test_find_missing_brackets_in_expr(self, _, expression, expected):
        self.assertEqual(expected, solutions.ParenthesisParser.find_missing_brackets_in_expr(expression))

    @parameterized.expand([
        ("own example 1", [
            "("
        ], 1),
        ("own example 2", [
            "["
        ], 2),
        ("own example 3", [
            "{"
        ], 3),
        ("own example 4", [
            "<"
        ], 4),
        ("own example 5", [
            "((",  # 6
            "[[",  # 12
            "{{",  # 18
            "<<",  # 24
            "(<"  # 21
        ], 18),
        ("aoc example", [
            "[({(<(())[]>[[{[]{<()<>>",
            "[(()[<>])]({[<{<<[]>>(",
            "{([(<{}[<>[]}>{[]{[(<()>",
            "(((({<>}<{<{<>}{[]{[]{}",
            "[[<[([]))<([[{}[[()]]]",
            "[{[{({}]{}}([{[{{{}}([]",
            "{<[[]]>}<{[{[{[]{()[[[]",
            "[<(<(<(<{}))><([]([]()",
            "<{([([[(<>()){}]>(<<{{",
            "<{([{{}}[<[[[<>{}]]]>[]]"
        ], 288957),
    ])
    def test_calculate_autocomplete_score(self, _, data, expected):
        parser = solutions.ParenthesisParser(data)
        self.assertEqual(expected, parser.calculate_autocomplete_score())


class Day11(unittest.TestCase):
    @parameterized.expand([
        ("aoc example 1", [
            [5, 4, 8, 3, 1, 4, 3, 2, 2, 3],
            [2, 7, 4, 5, 8, 5, 4, 7, 1, 1],
            [5, 2, 6, 4, 5, 5, 6, 1, 7, 3],
            [6, 1, 4, 1, 3, 3, 6, 1, 4, 6],
            [6, 3, 5, 7, 3, 8, 5, 4, 7, 8],
            [4, 1, 6, 7, 5, 2, 4, 6, 4, 5],
            [2, 1, 7, 6, 8, 4, 1, 7, 2, 1],
            [6, 8, 8, 2, 8, 8, 1, 1, 3, 4],
            [4, 8, 4, 6, 8, 4, 8, 5, 5, 4],
            [5, 2, 8, 3, 7, 5, 1, 5, 2, 6],
        ], 100, 1656),
        ("aoc example 2", [
            [1, 1, 1, 1, 1],
            [1, 9, 9, 9, 1],
            [1, 9, 1, 9, 1],
            [1, 9, 9, 9, 1],
            [1, 1, 1, 1, 1],
        ], 2, 9)
    ])
    def test_count_number_of_flashes(self, _, data, steps, expected):
        handler = solutions.OctopusEngeryHandler(data)
        self.assertEqual(expected, handler.count_number_of_flashes(steps))

    @parameterized.expand([
        ("aoc example 1", [
            [5, 4, 8, 3, 1, 4, 3, 2, 2, 3],
            [2, 7, 4, 5, 8, 5, 4, 7, 1, 1],
            [5, 2, 6, 4, 5, 5, 6, 1, 7, 3],
            [6, 1, 4, 1, 3, 3, 6, 1, 4, 6],
            [6, 3, 5, 7, 3, 8, 5, 4, 7, 8],
            [4, 1, 6, 7, 5, 2, 4, 6, 4, 5],
            [2, 1, 7, 6, 8, 4, 1, 7, 2, 1],
            [6, 8, 8, 2, 8, 8, 1, 1, 3, 4],
            [4, 8, 4, 6, 8, 4, 8, 5, 5, 4],
            [5, 2, 8, 3, 7, 5, 1, 5, 2, 6],
        ], 195),
    ])
    def test_count_number_of_flashes(self, _, data, expected):
        handler = solutions.OctopusEngeryHandler(data)
        self.assertEqual(expected, handler.count_steps_to_simultaneous_flash())


class Day12(unittest.TestCase):
    @parameterized.expand([
        ("aoc example 1", [
            ("start", "A"),
            ("start", "b"),
            ("A", "c"),
            ("A", "b"),
            ("b", "d"),
            ("A", "end"),
            ("b", "end")
        ], 10),
        ("aoc example 2", [
            ("dc", "end"),
            ("HN", "start"),
            ("start", "kj"),
            ("dc", "start"),
            ("dc", "HN"),
            ("LN", "dc"),
            ("HN", "end"),
            ("kj", "sa"),
            ("kj", "HN"),
            ("kj", "dc"),
        ], 19),
        ("aoc example 3", [
            ("fs", "end"),
            ("he", "DX"),
            ("fs", "he"),
            ("start", "DX"),
            ("pj", "DX"),
            ("end", "zg"),
            ("zg", "sl"),
            ("zg", "pj"),
            ("pj", "he"),
            ("RW", "he"),
            ("fs", "DX"),
            ("pj", "RW"),
            ("zg", "RW"),
            ("start", "pj"),
            ("he", "WI"),
            ("zg", "he"),
            ("pj", "fs"),
            ("start", "RW"),
        ], 226)
    ])
    def test_count_paths_visiting_all_small_caves_once(self, _, data, expected):
        pf = solutions.Pathfinder(data)
        self.assertEqual(expected, pf.count_paths_visiting_all_small_caves_once())

    @parameterized.expand([
        ("aoc example 1", [
            ("start", "A"),
            ("start", "b"),
            ("A", "c"),
            ("A", "b"),
            ("b", "d"),
            ("A", "end"),
            ("b", "end")
        ], 36),
        ("aoc example 2", [
            ("dc", "end"),
            ("HN", "start"),
            ("start", "kj"),
            ("dc", "start"),
            ("dc", "HN"),
            ("LN", "dc"),
            ("HN", "end"),
            ("kj", "sa"),
            ("kj", "HN"),
            ("kj", "dc"),
        ], 103),
        ("aoc example 3", [
            ("fs", "end"),
            ("he", "DX"),
            ("fs", "he"),
            ("start", "DX"),
            ("pj", "DX"),
            ("end", "zg"),
            ("zg", "sl"),
            ("zg", "pj"),
            ("pj", "he"),
            ("RW", "he"),
            ("fs", "DX"),
            ("pj", "RW"),
            ("zg", "RW"),
            ("start", "pj"),
            ("he", "WI"),
            ("zg", "he"),
            ("pj", "fs"),
            ("start", "RW"),
        ], 3509)
    ])
    def test_count_paths_visiting_all_small_caves_but_one_once(self, _, data, expected):
        pf = solutions.Pathfinder(data)
        self.assertEqual(expected, pf.count_paths_visiting_all_small_caves_but_one_once())


class Day13(unittest.TestCase):
    def setUp(self):
        self.dots = {
            (6, 10),
            (0, 14),
            (9, 10),
            (0, 3),
            (10, 4),
            (4, 11),
            (6, 0),
            (6, 12),
            (4, 1),
            (0, 13),
            (10, 12),
            (3, 4),
            (3, 0),
            (8, 4),
            (1, 10),
            (2, 14),
            (8, 10),
            (9, 0),
        }
        self.instructions = [
            ('y', 7),
            ('x', 5)
        ]

    def test_count_dots_after_single_fold(self):
        folder = solutions.OrigamiFolder((self.dots, self.instructions))
        expected = 17
        self.assertEqual(folder.count_dots_after_single_fold(), expected)

    def test_count_dots_after_full_fold(self):
        folder = solutions.OrigamiFolder((self.dots, self.instructions))
        expected = 16
        self.assertEqual(folder.count_dots_after_full_fold(), expected)


class Day14(unittest.TestCase):
    @parameterized.expand([
        ("aoc example 1", [
            "NNCB",
            ("CH", "B"),
            ("HH", "N"),
            ("CB", "H"),
            ("NH", "C"),
            ("HB", "C"),
            ("HC", "B"),
            ("HN", "C"),
            ("NN", "C"),
            ("BH", "H"),
            ("NC", "B"),
            ("NB", "B"),
            ("BN", "B"),
            ("BB", "N"),
            ("BC", "B"),
            ("CC", "N"),
            ("CN", "C")
        ], 10, 1588),
        ("aoc example 2", [
            "NNCB",
            ("CH", "B"),
            ("HH", "N"),
            ("CB", "H"),
            ("NH", "C"),
            ("HB", "C"),
            ("HC", "B"),
            ("HN", "C"),
            ("NN", "C"),
            ("BH", "H"),
            ("NC", "B"),
            ("NB", "B"),
            ("BN", "B"),
            ("BB", "N"),
            ("BC", "B"),
            ("CC", "N"),
            ("CN", "C")
        ], 40, 2188189693529)
    ])
    def test_most_common_least_common_diff(self, _, data, steps, expected):
        handler = solutions.PolymerHandler(data)
        self.assertEqual(expected, handler.most_common_least_common_diff(steps))


class Day15(unittest.TestCase):
    @parameterized.expand([
        ("aoc example 1", [
            [1, 1, 6, 3, 7, 5, 1, 7, 4, 2],
            [1, 3, 8, 1, 3, 7, 3, 6, 7, 2],
            [2, 1, 3, 6, 5, 1, 1, 3, 2, 8],
            [3, 6, 9, 4, 9, 3, 1, 5, 6, 9],
            [7, 4, 6, 3, 4, 1, 7, 1, 1, 1],
            [1, 3, 1, 9, 1, 2, 8, 1, 3, 7],
            [1, 3, 5, 9, 9, 1, 2, 4, 2, 1],
            [3, 1, 2, 5, 4, 2, 1, 6, 3, 9],
            [1, 2, 9, 3, 1, 3, 8, 5, 2, 1],
            [2, 3, 1, 1, 9, 4, 4, 5, 8, 1],
        ], 40),
        ("own example 1", [
            [1, 9, 6, 3, 7, 5, 1, 7, 4, 2],
            [1, 9, 8, 1, 3, 7, 3, 6, 7, 2],
            [1, 9, 3, 6, 5, 1, 1, 3, 2, 8],
            [1, 9, 9, 4, 9, 3, 1, 5, 6, 9],
            [1, 9, 6, 3, 4, 1, 7, 1, 1, 1],
            [1, 9, 1, 9, 1, 2, 8, 1, 3, 7],
            [1, 9, 5, 9, 9, 1, 2, 4, 2, 1],
            [1, 9, 2, 5, 4, 2, 1, 6, 3, 9],
            [1, 9, 9, 3, 1, 3, 8, 5, 2, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ], 18),
        ("own example 2", [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ], 18),
        ("own example 2", [
            [1, 1, 1, 1, 1, 8, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 9, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 9, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 8, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 9, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 9, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 9, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 9, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 9, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 9, 1, 1, 1, 1],
        ], 25)
    ])
    def test_find_path_of_the_lowest_risk(self, _, data, expected):
        pathfinder = solutions.WeightedPathfinder(data)
        self.assertEqual(expected, pathfinder.find_path_of_the_lowest_risk())

    @parameterized.expand([
        ("own example 1", [
            [1, 1],
            [1, 1],
        ], [
             [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
             [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
             [2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
             [2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
             [3, 3, 4, 4, 5, 5, 6, 6, 7, 7],
             [3, 3, 4, 4, 5, 5, 6, 6, 7, 7],
             [4, 4, 5, 5, 6, 6, 7, 7, 8, 8],
             [4, 4, 5, 5, 6, 6, 7, 7, 8, 8],
             [5, 5, 6, 6, 7, 7, 8, 8, 9, 9],
             [5, 5, 6, 6, 7, 7, 8, 8, 9, 9]
         ]),
        ("own example 2", [
            [1, 2],
            [3, 4],
        ], [
             [1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
             [3, 4, 4, 5, 5, 6, 6, 7, 7, 8],
             [2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
             [4, 5, 5, 6, 6, 7, 7, 8, 8, 9],
             [3, 4, 4, 5, 5, 6, 6, 7, 7, 8],
             [5, 6, 6, 7, 7, 8, 8, 9, 9, 1],
             [4, 5, 5, 6, 6, 7, 7, 8, 8, 9],
             [6, 7, 7, 8, 8, 9, 9, 1, 1, 2],
             [5, 6, 6, 7, 7, 8, 8, 9, 9, 1],
             [7, 8, 8, 9, 9, 1, 1, 2, 2, 3]
         ])
    ])
    def test_extend_data(self, _, data, expected):
        extended_data = solutions.WeightedPathfinder.extend_data(data)
        # for x in extended_data:
        #     print(x)
        self.assertEqual(expected, extended_data)

    @parameterized.expand([
        ("aoc example 1", [
            [1, 1, 6, 3, 7, 5, 1, 7, 4, 2],
            [1, 3, 8, 1, 3, 7, 3, 6, 7, 2],
            [2, 1, 3, 6, 5, 1, 1, 3, 2, 8],
            [3, 6, 9, 4, 9, 3, 1, 5, 6, 9],
            [7, 4, 6, 3, 4, 1, 7, 1, 1, 1],
            [1, 3, 1, 9, 1, 2, 8, 1, 3, 7],
            [1, 3, 5, 9, 9, 1, 2, 4, 2, 1],
            [3, 1, 2, 5, 4, 2, 1, 6, 3, 9],
            [1, 2, 9, 3, 1, 3, 8, 5, 2, 1],
            [2, 3, 1, 1, 9, 4, 4, 5, 8, 1],
        ], 315)
    ])
    def test_find_path_of_the_lowest_risk_extended(self, _, data, expected):
        pathfinder = solutions.WeightedPathfinder(data)
        self.assertEqual(expected, pathfinder.find_path_of_the_lowest_risk_extended())


class Day16(unittest.TestCase):
    @parameterized.expand([
        ("aoc example 1", "D2FE28", 6),
        ("aoc example 2", "38006F45291200", 9),
        ("aoc example 3", "EE00D40C823060", 14),
        ("aoc example 4", "8A004A801A8002F478", 16),
        ("aoc example 5", "620080001611562C8802118E34", 12),
        ("aoc example 6", "C0015000016115A2E0802F182340", 23),
        ("aoc example 7", "A0016C880162017C3686B18A3D4780", 31),
    ])
    def test_calculate_sum_of_packet_versions(self, _, data, expected):
        handler = solutions.TransmissionHandler(data)
        self.assertEqual(expected, handler.calculate_sum_of_packet_versions())

    @parameterized.expand([
        ("aoc example 1", "C200B40A82", 3),
        ("aoc example 2", "04005AC33890", 54),
        ("aoc example 3", "880086C3E88112", 7),
        ("aoc example 4", "CE00C43D881120", 9),
        ("aoc example 5", "D8005AC2A8F0", 1),
        ("aoc example 6", "F600BC2D8F", 0),
        ("aoc example 7", "9C005AC2F8F0", 0),
        ("aoc example 8", "9C0141080250320F1802104A08", 1),
    ])
    def test_calculate_value_of_the_outermost_packet(self, _, data, expected):
        handler = solutions.TransmissionHandler(data)
        self.assertEqual(expected, handler.calculate_value_of_the_outermost_packet())


class Day17(unittest.TestCase):
    @parameterized.expand([
        ("aoc example 1", (20, 30, -10, -5), 45),
    ])
    def test_find_highest_y(self, _, data, expected):
        launcher = solutions.ProbeLauncher(data)
        launcher.compute()
        self.assertEqual(expected, launcher.find_highest_y())

    @parameterized.expand([
        ("aoc example 1", (20, 30, -10, -5), 112),
    ])
    def test_find_highest_y(self, _, data, expected):
        launcher = solutions.ProbeLauncher(data)
        launcher.compute()
        self.assertEqual(expected, launcher.count_initial_velocities())


class Day18(unittest.TestCase):
    @parameterized.expand([
        ("aoc example 1", [[9, 4], [8, 4], [1, 3], [2, 2], [3, 1], [4, 0]],
         [[0, 3], [9, 3], [2, 2], [3, 1], [4, 0]]),
        ("aoc example 2", [[7, 0], [6, 1], [5, 2], [4, 3], [3, 4], [2, 4]],
         [[7, 0], [6, 1], [5, 2], [7, 3], [0, 3]]),
        ("aoc example 3", [[6, 1], [5, 2], [4, 3], [3, 4], [2, 4], [1, 0]],
         [[6, 1], [5, 2], [7, 3], [0, 3], [3, 0]]),
        ("aoc example 4", [[3, 1], [2, 2], [1, 3], [7, 4], [3, 4], [6, 1], [5, 2], [4, 3], [3, 4], [2, 4]],
         [[3, 1], [2, 2], [8, 3], [0, 3], [9, 1], [5, 2], [4, 3], [3, 4], [2, 4]]),
        ("aoc example 5", [[3, 1], [2, 2], [8, 3], [0, 3], [9, 1], [5, 2], [4, 3], [3, 4], [2, 4]],
         [[3, 1], [2, 2], [8, 3], [0, 3], [9, 1], [5, 2], [7, 3], [0, 3]])
    ])
    def test_single_explode_performed(self, _, data, expected):
        self.assertEqual((True, expected), solutions.SnailfishCalculator.explode(data))

    @parameterized.expand([
        ("aoc example 1", [[1, 0], [1, 0]],
         [[1, 0], [1, 0]]),
        ("aoc example 2", [[1, 0], [1, 0], [9, 1], [9, 1]],
         [[1, 0], [1, 0], [9, 1], [9, 1]]),
    ])
    def test_single_explode_not_performed(self, _, data, expected):
        self.assertEqual((False, expected), solutions.SnailfishCalculator.explode(data))

    @parameterized.expand([
        ("aoc example 1", [[0, 3], [7, 3], [4, 2], [15, 2], [0, 3], [13, 3], [1, 1], [1, 1]],
         [[0, 3], [7, 3], [4, 2], [7, 3], [8, 3], [0, 3], [13, 3], [1, 1], [1, 1]]),
        ("aoc example 2", [[0, 3], [7, 3], [4, 2], [7, 3], [8, 3], [0, 3], [13, 3], [1, 1], [1, 1]],
         [[0, 3], [7, 3], [4, 2], [7, 3], [8, 3], [0, 3], [6, 4], [7, 4], [1, 1], [1, 1]]),
    ])
    def test_single_split_performed(self, _, data, expected):
        self.assertEqual((True, expected), solutions.SnailfishCalculator.split(data))

    @parameterized.expand([
        ("aoc example 1", [[1, 0], [1, 0]],
         [[1, 0], [1, 0]]),
        ("aoc example 2", [[1, 0], [1, 0], [9, 1], [9, 1]],
         [[1, 0], [1, 0], [9, 1], [9, 1]]),
    ])
    def test_single_split_not_performed(self, _, data, expected):
        self.assertEqual((False, expected), solutions.SnailfishCalculator.split(data))

    @parameterized.expand([
        ("aoc example 1", [[1, 0], [2, 0]],
         [[3, 1], [4, 1], [5, 0]],
         [[1, 1], [2, 1], [3, 2], [4, 2], [5, 1]])
    ])
    def test_single_add(self, _, num1, num2, expected):
        self.assertEqual(expected, solutions.SnailfishCalculator.add(num1, num2))

    @parameterized.expand([
        ("aoc example 1", [[[1, 0], [1, 0]],
                           [[2, 0], [2, 0]],
                           [[3, 0], [3, 0]],
                           [[4, 0], [4, 0]]],
         [[1, 3], [1, 3], [2, 3], [2, 3], [3, 2], [3, 2], [4, 1], [4, 1]]),
        ("aoc example 2", [[[1, 0], [1, 0]],
                           [[2, 0], [2, 0]],
                           [[3, 0], [3, 0]],
                           [[4, 0], [4, 0]],
                           [[5, 0], [5, 0]]],
         [[3, 3], [0, 3], [5, 3], [3, 3], [4, 2], [4, 2], [5, 1], [5, 1]]),
        ("aoc example 3", [[[1, 0], [1, 0]],
                           [[2, 0], [2, 0]],
                           [[3, 0], [3, 0]],
                           [[4, 0], [4, 0]],
                           [[5, 0], [5, 0]],
                           [[6, 0], [6, 0]]],
         [[5, 3], [0, 3], [7, 3], [4, 3], [5, 2], [5, 2], [6, 1], [6, 1]]),
        ("aoc example 4", [[[0, 2], [4, 3], [5, 3], [0, 2], [0, 2], [4, 3], [5, 3], [2, 3], [6, 3], [9, 2], [5, 2]],
                           [[7, 0], [3, 3], [7, 3], [4, 3], [3, 3], [6, 3], [3, 3], [8, 3], [8, 3]],
                           [[2, 1], [0, 3], [8, 3], [3, 3], [4, 3], [6, 3], [7, 3], [1, 2], [7, 2], [1, 3], [6, 3]]],
         [[6, 3], [7, 3], [6, 3], [7, 3], [7, 3], [7, 3], [0, 3], [7, 3], [8, 3], [7, 3], [7, 3], [7, 3], [8, 3],
          [8, 3], [8, 3], [0, 3]])
    ])
    def test_chain_add(self, _, numbers, expected):
        self.assertEqual(expected, solutions.SnailfishCalculator.chain_add(numbers))

    @parameterized.expand([
        ("aoc example 1", [[9, 0], [1, 0]], 29),
        ("aoc example 2", [[1, 0], [9, 0]], 21),
        ("aoc example 3", [[9, 1], [1, 1], [1, 1], [9, 1]], 129),
        ("aoc example 4", [[1, 1], [2, 1], [3, 2], [4, 2], [5, 1]], 143),
        ("aoc example 5", [[0, 3], [7, 3], [4, 2], [7, 3], [8, 3], [6, 3], [0, 3], [8, 1], [1, 1]], 1384)
    ])
    def test_calculate_magnitude(self, _, data, expected):
        self.assertEqual(expected, solutions.SnailfishCalculator.calculate_magnitude(data))

    @parameterized.expand([
        ("aoc example 1", [
            [[0, 2], [5, 3], [8, 3], [1, 3], [7, 3], [9, 3], [6, 3], [4, 2], [1, 3], [2, 3], [1, 3], [4, 3], [2, 2]],
            [[5, 2], [2, 3], [8, 3], [4, 1], [5, 1], [9, 3], [9, 3], [0, 2]],
            [[6, 0], [6, 3], [2, 3], [5, 3], [6, 3], [7, 3], [6, 3], [4, 3], [7, 3]],
            [[6, 2], [0, 3], [7, 3], [0, 2], [9, 2], [4, 1], [9, 2], [9, 3], [0, 3]],
            [[7, 2], [6, 3], [4, 3], [3, 2], [1, 3], [3, 3], [5, 3], [5, 3], [1, 2], [9, 1]],
            [[6, 1], [7, 3], [3, 3], [3, 3], [2, 3], [3, 3], [8, 3], [5, 3], [7, 3], [4, 1]],
            [[5, 3], [4, 3], [7, 3], [7, 3], [8, 1], [8, 2], [3, 2], [8, 1]],
            [[9, 1], [3, 1], [9, 2], [9, 2], [6, 2], [4, 3], [9, 3]],
            [[2, 1], [7, 3], [7, 3], [7, 2], [5, 2], [8, 2], [9, 3], [3, 3], [0, 3], [2, 3]],
            [[5, 3], [2, 3], [5, 2], [8, 2], [3, 3], [7, 3], [5, 2], [7, 3], [5, 3], [4, 2], [4, 2]]
        ], 3993),
    ])
    def test_calculate_largest_sum_of_two_numbers(self, _, data, expected):
        calc = solutions.SnailfishCalculator(data)
        self.assertEqual(expected, calc.calculate_largest_sum_of_two_numbers())


class Day19(unittest.TestCase):
    @parameterized.expand([
        ("aoc example 1", [
            [
                [404, -588, -901],
                [528, -643, 409],
                [-838, 591, 734],
                [390, -675, -793],
                [-537, -823, -458],
                [-485, -357, 347],
                [-345, -311, 381],
                [-661, -816, -575],
                [-876, 649, 763],
                [-618, -824, -621],
                [553, 345, -567],
                [474, 580, 667],
                [-447, -329, 318],
                [-584, 868, -557],
                [544, -627, -890],
                [564, 392, -477],
                [455, 729, 728],
                [-892, 524, 684],
                [-689, 845, -530],
                [423, -701, 434],
                [7, -33, -71],
                [630, 319, -379],
                [443, 580, 662],
                [-789, 900, -551],
                [459, -707, 401],
            ],
            [
                [686, 422, 578],
                [605, 423, 415],
                [515, 917, -361],
                [-336, 658, 858],
                [95, 138, 22],
                [-476, 619, 847],
                [-340, -569, -846],
                [567, -361, 727],
                [-460, 603, -452],
                [669, -402, 600],
                [729, 430, 532],
                [-500, -761, 534],
                [-322, 571, 750],
                [-466, -666, -811],
                [-429, -592, 574],
                [-355, 545, -477],
                [703, -491, -529],
                [-328, -685, 520],
                [413, 935, -424],
                [-391, 539, -444],
                [586, -435, 557],
                [-364, -763, -893],
                [807, -499, -711],
                [755, -354, -619],
                [553, 889, -390]
            ],
            [
                [649, 640, 665],
                [682, -795, 504],
                [-784, 533, -524],
                [-644, 584, -595],
                [-588, -843, 648],
                [-30, 6, 44],
                [-674, 560, 763],
                [500, 723, -460],
                [609, 671, -379],
                [-555, -800, 653],
                [-675, -892, -343],
                [697, -426, -610],
                [578, 704, 681],
                [493, 664, -388],
                [-671, -858, 530],
                [-667, 343, 800],
                [571, -461, -707],
                [-138, -166, 112],
                [-889, 563, -600],
                [646, -828, 498],
                [640, 759, 510],
                [-630, 509, 768],
                [-681, -892, -333],
                [673, -379, -804],
                [-742, -814, -386],
                [577, -820, 562]
            ],
            [
                [-589, 542, 597],
                [605, -692, 669],
                [-500, 565, -823],
                [-660, 373, 557],
                [-458, -679, -417],
                [-488, 449, 543],
                [-626, 468, -788],
                [338, -750, -386],
                [528, -832, -391],
                [562, -778, 733],
                [-938, -730, 414],
                [543, 643, -506],
                [-524, 371, -870],
                [407, 773, 750],
                [-104, 29, 83],
                [378, -903, -323],
                [-778, -728, 485],
                [426, 699, 580],
                [-438, -605, -362],
                [-469, -447, -387],
                [509, 732, 623],
                [647, 635, -688],
                [-868, -804, 481],
                [614, -800, 639],
                [595, 780, -596]
            ],
            [
                [727, 592, 562],
                [-293, -554, 779],
                [441, 611, -461],
                [-714, 465, -776],
                [-743, 427, -804],
                [-660, -479, -426],
                [832, -632, 460],
                [927, -485, -438],
                [408, 393, -506],
                [466, 436, -512],
                [110, 16, 151],
                [-258, -428, 682],
                [-393, 719, 612],
                [-211, -452, 876],
                [808, -476, -593],
                [-575, 615, 604],
                [-485, 667, 467],
                [-680, 325, -822],
                [-627, -443, -432],
                [872, -547, -609],
                [833, 512, 582],
                [807, 604, 487],
                [839, -516, 451],
                [891, -625, 532],
                [-652, -548, -490],
                [30, -46, -14]
            ]
        ], 79),
    ])
    def test_count_beacons(self, _, data, expected):
        handler = solutions.BeaconHandler(data)
        self.assertEqual(expected, handler.count_beacons())

    @parameterized.expand([
        ("rotate x1", [1, 2, 3], [0], [1, 3, -2]),
        ("rotate x2", [1, 2, 3], [1], [1, -2, -3]),
        ("rotate x3", [1, 2, 3], [2], [1, -3, 2]),

        ("rotate y1", [1, 2, 3], [3], [-3, 2, 1]),
        ("rotate y2", [1, 2, 3], [4], [-1, 2, -3]),
        ("rotate y3", [1, 2, 3], [5], [3, 2, -1]),

        ("rotate z1", [1, 2, 3], [6], [2, -1, 3]),
        ("rotate z2", [1, 2, 3], [7], [-1, -2, 3]),
        ("rotate z3", [1, 2, 3], [8], [-2, 1, 3]),
    ])
    def test_rotate(self, _, point, rot, expected):
        self.assertEqual(expected, solutions.BeaconHandler.rotate(point, rot))

    @parameterized.expand([
        ("single point", [[1, 2, 3]], {
            (-1, -1): [[1, 2, 3]],
            (-1, 3): [[-3, 2, 1]],
            (-1, 4): [[-1, 2, -3]],
            (-1, 5): [[3, 2, -1]],

            (0, -1): [[1, 3, -2]],
            (0, 3): [[2, 3, 1]],
            (0, 4): [[-1, 3, 2]],
            (0, 5): [[-2, 3, -1]],

            (1, -1): [[1, -2, -3]],
            (1, 3): [[3, -2, 1]],
            (1, 4): [[-1, -2, 3]],
            (1, 5): [[-3, -2, -1]],

            (2, -1): [[1, -3, 2]],
            (2, 3): [[-2, -3, 1]],
            (2, 4): [[-1, -3, -2]],
            (2, 5): [[2, -3, -1]],

            (6, -1): [[2, -1, 3]],
            (6, 3): [[-3, -1, 2]],
            (6, 4): [[-2, -1, -3]],
            (6, 5): [[3, -1, -2]],

            (8, -1): [[-2, 1, 3]],
            (8, 3): [[-3, 1, -2]],
            (8, 4): [[2, 1, -3]],
            (8, 5): [[3, 1, 2]],
        }),

    ])
    def test_generate_all_rotations(self, _, points, expected):
        self.assertEqual(expected, solutions.BeaconHandler.generate_all_rotations(points))

    @parameterized.expand([
        ("aoc example 1", [1105, -1205, 1229], [-92, -2380, -20], 3621),
    ])
    def test_calculate_manhattan_distance(self, _, point_a, point_b, expected):
        self.assertEqual(expected, solutions.BeaconHandler.calculate_manhattan_distance(point_a, point_b))


class Day20(unittest.TestCase):
    @parameterized.expand([
        ("own example 1", [
            ['#', '#', '.'],
            ['#', '.', '.'],
            ['#', '#', '#']
        ], '.', 1, [
            ['.', '.', '.', '.', '.'],
            ['.', '#', '#', '.', '.'],
            ['.', '#', '.', '.', '.'],
            ['.', '#', '#', '#', '.'],
            ['.', '.', '.', '.', '.']
        ])
    ])
    def test_apply_padding(self, _, img, padding_char, padding_size, expected):
        self.assertEqual(expected, solutions.ImageEnhancer.apply_padding(img, padding_char, padding_size))


class Day21(unittest.TestCase):
    @parameterized.expand([
        ("aoc example 1", (4, 8), 100, 739785),
    ])
    def test_calculate_losing_score_x_num_of_rolls(self, _, players_pos, dice_size, expected):
        roller = solutions.DiceRoller(players_pos)
        self.assertEqual(expected, roller.calculate_losing_score_x_num_of_rolls(dice_size))

    @parameterized.expand([
        ("aoc example 1", (4, 8), 3, 444356092776315),
    ])
    def test_calculate_number_of_universes(self, _, players_pos, dice_size, expected):
        roller = solutions.DiceRoller(players_pos)
        self.assertEqual(expected, roller.calculate_number_of_universes(dice_size))
