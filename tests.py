import solutions
import day1
import day2
import day3
import day4
import day5
import day6
import day7
import day8
import day9
import day10
import unittest
from parameterized import parameterized


class Day1(unittest.TestCase):
    @parameterized.expand([
        ("example 1", [], 0),
        ("example 2", [199, 200, 208, 210, 200, 207, 240, 269, 260, 263], 7),
        ("example 3 puzzle data", day1.parse_day1_data(), 1121),
    ])
    def test_count_number_of_increases(self, _, data, expected):
        self.assertEqual(expected, day1.count_number_of_increases(data))

    @parameterized.expand([
        ("example 1", [], 0),
        ("example 2", [1], 0),
        ("example 3", [1, 10], 0),
        ("example 4", [1, 10, 100], 0),
        ("example 5", [199, 200, 208, 210, 200, 207, 240, 269, 260, 263], 5),
        ("example 6 puzzle data", day1.parse_day1_data(), 1065),
    ])
    def test_count_number_of_increased_windows(self, _, data, expected):
        self.assertEqual(expected, day1.count_number_of_increased_windows(data))


class Day2(unittest.TestCase):
    data = [
        ("forward", "5"),
        ("down", "5"),
        ("forward", "8"),
        ("up", "3"),
        ("down", "8"),
        ("forward", "2")
    ]

    @parameterized.expand([
        ("example 1", data, 150),
        ("example 2 puzzle data", day2.parse_day2_data(), 1924923),
    ])
    def test_calculate_hrz_depth_product(self, _, data, expected):
        self.assertEqual(expected, day2.calculate_hrz_depth_product(data))

    @parameterized.expand([
        ("example 1", data, 900),
        ("example 2 puzzle data", day2.parse_day2_data(), 1982495697),
    ])
    def test_calculate_hrz_depth_aim_product(self, _, data, expected):
        self.assertEqual(expected, day2.calculate_hrz_depth_aim_product(data))


class Day3(unittest.TestCase):
    data = [
             '00100',
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
             '01010'
    ]

    @parameterized.expand([
        ("example 1", data, 198),
        ("example 2 puzzle data", day3.parse_day3_data(), 4139586),
    ])
    def test_calculate_power_consumption(self, _, data, expected):
        self.assertEqual(expected, day3.calculate_power_consumption(data))

    @parameterized.expand([
        ("example 1", data, 230),
        ("example 2 puzzle data", day3.parse_day3_data(), 1800151),
    ])
    def test_calculate_life_support_rating(self, _, data, expected):
        self.assertEqual(expected, day3.calculate_life_support_rating(data))


class Day4(unittest.TestCase):
    data = [
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

    @parameterized.expand([
        ("example 1", data,
         [7, 4, 9, 5, 11, 17, 23, 2, 0, 14, 21, 24, 10, 16, 13, 6, 15, 25, 12, 22, 18, 20, 8, 19, 3, 26, 1], 4512),
        ("example 2 puzzle data", day4.parse_day4_data()[0], day4.parse_day4_data()[1], 89001),
    ])
    def test_get_first_winner_points(self, _, boards, nums, expected):
        bingo = day4.Bingo(boards, nums)
        while bingo.is_move_possible():
            bingo.step()

        self.assertEqual(expected, bingo.get_winner_points())

    @parameterized.expand([
        ("example 1", data,
         [7, 4, 9, 5, 11, 17, 23, 2, 0, 14, 21, 24, 10, 16, 13, 6, 15, 25, 12, 22, 18, 20, 8, 19, 3, 26, 1], 1924),
        ("example 2 puzzle data", day4.parse_day4_data()[0], day4.parse_day4_data()[1], 7296),
    ])
    def test_get_last_winner_points(self, _, boards, nums, expected):
        bingo = day4.AntiBingo(boards, nums)
        while bingo.is_move_possible():
            bingo.step()

        self.assertEqual(expected, bingo.get_winner_points())


class Day5(unittest.TestCase):
    data = [
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

    @parameterized.expand([
        ("example 1", data, 5),
        ("example 2 puzzle data", day5.parse_day5_data(), 7142),
    ])
    def test_count_overlapping_horizontal_vertical(self, _, data, expected):
        self.assertEqual(expected, day5.count_overlapping_horizontal_vertical(data))

    @parameterized.expand([
        ("example 1", data, 12),
        ("example 2 puzzle data", day5.parse_day5_data(), 20012),
    ])
    def test_count_overlapping_horizontal_vertical_diagonal(self, _, data, expected):
        self.assertEqual(expected, day5.count_overlapping_horizontal_vertical_diagonal(data))


class Day6(unittest.TestCase):
    @parameterized.expand([
        ("example 1", [3, 4, 3, 1, 2], 80, 5934),
        ("example 2 puzzle part 1 data", day6.parse_day6_data(), 80, 358214),
        ("example 3 puzzle part 2 data", day6.parse_day6_data(), 256, 1622533344325),
    ])
    def test_count_lanternfish(self, _, data, days, expected):
        self.assertEqual(expected, day6.count_lanternfish(data, days))


class Day7(unittest.TestCase):
    @parameterized.expand([
        ("example 1 const fuel usage", [16, 1, 2, 0, 4, 2, 7, 1, 2, 14], 37, lambda x: x),
        ("example 2 increasing fuel usage", [16, 1, 2, 0, 4, 2, 7, 1, 2, 14], 168, lambda x: int(0.5 * x * (x + 1))),
        ("example 3 puzzle part 1 data", day7.parse_day7_data(), 355150, lambda x: x),
        ("example 4 puzzle part 2 data", day7.parse_day7_data(), 98368490, lambda x: int(0.5 * x * (x + 1))),
    ])
    def test_find_optimal_fuel_usage(self, _, data, expected, increase):
        self.assertEqual(expected, day7.find_optimal_fuel_usage(data, increase))


class Day8(unittest.TestCase):
    raw_data = [
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

    @parameterized.expand([
        ("example 1", [[x.split() for x in line.split(" | ")] for line in raw_data], 26),
        ("example 2 puzzle data", day8.parse_day8_data(), 397),
    ])
    def test_count_digits_made_of_unique_number_of_segments(self, _, data, expected):
        self.assertEqual(expected, day8.count_digits_made_of_unique_number_of_segments(data))

    @parameterized.expand([
        ("example 1", [[x.split() for x in line.split(" | ")] for line in raw_data], 61229),
        ("example 2 puzzle data", day8.parse_day8_data(), 1027422),
    ])
    def test_calculate_sum_of_all_output_digits(self, _, data, expected):
        self.assertEqual(expected, day8.calculate_sum_of_all_output_digits(data))


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
        ("example 3 puzzle data", day9.parse_day9_data(), 506),
    ])
    def test_calculate_sum_of_the_risk_levels(self, _, data, expected):
        self.assertEqual(expected, day9.calculate_sum_of_the_risk_levels(data))

    @parameterized.expand([
        ("aoc example", [
            [2, 1, 9, 9, 9, 4, 3, 2, 1, 0],
            [3, 9, 8, 7, 8, 9, 4, 9, 2, 1],
            [9, 8, 5, 6, 7, 8, 9, 8, 9, 2],
            [8, 7, 6, 7, 8, 9, 6, 7, 8, 9],
            [9, 8, 9, 9, 9, 6, 5, 6, 7, 8]
        ], 1134),
        ("example 2 puzzle data", day9.parse_day9_data(), 931200),
    ])
    def test_calculate_sum_of_three_largest_basins(self, _, data, expected):
        basin_handler = day9.BasinHandler(data)
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
        self.assertEqual(expected, day10.ParenthesisParser.find_first_corrupted_bracket_in_expr(expression))

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
        ("puzzle data", day10.parse_day10_data(), 390993)
    ])
    def test_calculate_syntax_error_score(self, _, data, expected):
        parser = day10.ParenthesisParser(data)
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
        self.assertEqual(expected, day10.ParenthesisParser.find_missing_brackets_in_expr(expression))

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
        ("puzzle data", day10.parse_day10_data(), 2391385187)
    ])
    def test_calculate_autocomplete_score(self, _, data, expected):
        parser = day10.ParenthesisParser(data)
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
        ("aoc example 1", (4, 8), 444356092776315),
    ])
    def test_calculate_number_of_universes(self, _, players_pos, expected):
        roller = solutions.DiceRoller(players_pos)
        self.assertEqual(expected, roller.calculate_number_of_universes())


class Day22(unittest.TestCase):
    @parameterized.expand([
        ("aoc example 1", [
            (True, 10, 12, 10, 12, 10, 12),
            (True, 11, 13, 11, 13, 11, 13),
            (False, 9, 11, 9, 11, 9, 11),
            (True, 10, 10, 10, 10, 10, 10),
        ], 39),
    ])
    def test_count_cubes_in_initialization_area(self, _, data, expected):
        handler = solutions.ReactorHandler(data)
        self.assertEqual(expected, handler.count_cubes_in_initialization_area())

    @parameterized.expand([
        ("own example 1 empty same size",
         (10, 12, 10, 12, 10, 12),
         (10, 12, 10, 12, 10, 12),
         []),
        ("own example 2 L",
         (10, 12, 10, 12, 10, 12),
         (11, 12, 10, 12, 10, 12),
         [(10, 11, 10, 12, 10, 12)]),
        ("own example 3 LRFBUD",
         (0, 6, 0, 4, -3, 0),
         (2, 4, 2, 4, -2, -1),
         [(0, 2, 0, 4, -3, 0),  # L
          (4, 6, 0, 4, -3, 0),  # R
          (2, 4, 0, 4, -1, 0),  # F
          (2, 4, 0, 4, -3, -2),  # B
          (2, 4, 0, 2, -2, -1)]),  # D
        ("own example 4 empty bigger",
         (10, 12, 10, 12, 10, 12),
         (0, 100, 1, 120, -2000, 1244),
         []),
        ("own example 5 R",
         (10, 12, 10, 12, 10, 12),
         (0, 11, -100, 1200, -2000, 1244),
         [(11, 12, 10, 12, 10, 12)]),
        ("own example 6 U",
         (0, 5, 0, 5, 0, 5),
         (0, 5, 0, 4, 0, 5),
         [(0, 5, 4, 5, 0, 5)]),
        ("own example 7 D",
         (0, 5, 0, 5, 0, 5),
         (0, 5, 1, 5, 0, 5),
         [(0, 5, 0, 1, 0, 5)]),
        ("own example 8 F",
         (0, 5, 0, 5, 0, 5),
         (0, 5, 0, 5, 0, 4),
         [(0, 5, 0, 5, 4, 5)]),
        ("own example 9 B",
         (0, 5, 0, 5, 0, 5),
         (0, 5, 0, 5, 1, 5),
         [(0, 5, 0, 5, 0, 1)]),
        ("own example 10",
         (-2, 3, -2, 3, 2, 3),
         (-2, 2, -1, 4, -3, 0),
         [(-2, 3, -2, 3, 2, 3)]),
    ])
    def test_cuboid_diff(self, _, a, b, expected):
        self.assertEqual(expected, solutions.ReactorHandler.cuboid_diff(a, b))

    @parameterized.expand([
        ("own example 1", (10, 12, 10, 12, 10, 12), 8),
        ("own example 2", (-10, 12, -10, 12, 10, 120), 53240),
        ("own example 2", (-10, -10, -10, 12, 10, 120), 0),
    ])
    def test_cuboid_volume(self, _, cuboid, expected):
        self.assertEqual(expected, solutions.ReactorHandler.cuboid_volume(cuboid))

    @parameterized.expand([
        ("aoc example 1 from part 2", [
            (True, -5, 47, -31, 22, -19, 33),
            (True, -44, 5, -27, 21, -14, 35),
            (True, -49, -1, -11, 42, -10, 38),
            (True, -20, 34, -40, 6, -44, 1),
            (False, 26, 39, 40, 50, -2, 11),
            (True, -41, 5, -41, 6, -36, 8),
            (False, -43, -33, -45, -28, 7, 25),
            (True, -33, 15, -32, 19, -34, 11),
            (False, 35, 47, -46, -34, -11, 5),
            (True, -14, 36, -6, 44, -16, 29),
            (True, -57795, -6158, 29564, 72030, 20435, 90618),
            (True, 36731, 105352, -21140, 28532, 16094, 90401),
            (True, 30999, 107136, -53464, 15513, 8553, 71215),
            (True, 13528, 83982, -99403, -27377, -24141, 23996),
            (True, -72682, -12347, 18159, 111354, 7391, 80950),
            (True, -1060, 80757, -65301, -20884, -103788, -16709),
            (True, -83015, -9461, -72160, -8347, -81239, -26856),
            (True, -52752, 22273, -49450, 9096, 54442, 119054),
            (True, -29982, 40483, -108474, -28371, -24328, 38471),
            (True, -4958, 62750, 40422, 118853, -7672, 65583),
            (True, 55694, 108686, -43367, 46958, -26781, 48729),
            (True, -98497, -18186, -63569, 3412, 1232, 88485),
            (True, -726, 56291, -62629, 13224, 18033, 85226),
            (True, -110886, -34664, -81338, -8658, 8914, 63723),
            (True, -55829, 24974, -16897, 54165, -121762, -28058),
            (True, -65152, -11147, 22489, 91432, -58782, 1780),
            (True, -120100, -32970, -46592, 27473, -11695, 61039),
            (True, -18631, 37533, -124565, -50804, -35667, 28308),
            (True, -57817, 18248, 49321, 117703, 5745, 55881),
            (True, 14781, 98692, -1341, 70827, 15753, 70151),
            (True, -34419, 55919, -19626, 40991, 39015, 114138),
            (True, -60785, 11593, -56135, 2999, -95368, -26915),
            (True, -32178, 58085, 17647, 101866, -91405, -8878),
            (True, -53655, 12091, 50097, 105568, -75335, -4862),
            (True, -111166, -40997, -71714, 2688, 5609, 50954),
            (True, -16602, 70118, -98693, -44401, 5197, 76897),
            (True, 16383, 101554, 4615, 83635, -44907, 18747),
            (False, -95822, -15171, -19987, 48940, 10804, 104439),
            (True, -89813, -14614, 16069, 88491, -3297, 45228),
            (True, 41075, 99376, -20427, 49978, -52012, 13762),
            (True, -21330, 50085, -17944, 62733, -112280, -30197),
            (True, -16478, 35915, 36008, 118594, -7885, 47086),
            (False, -98156, -27851, -49952, 43171, -99005, -8456),
            (False, 2032, 69770, -71013, 4824, 7471, 94418),
            (True, 43670, 120875, -42068, 12382, -24787, 38892),
            (False, 37514, 111226, -45862, 25743, -16714, 54663),
            (False, 25699, 97951, -30668, 59918, -15349, 69697),
            (False, -44271, 17935, -9516, 60759, 49131, 112598),
            (True, -61695, -5813, 40978, 94975, 8655, 80240),
            (False, -101086, -9439, -7088, 67543, 33935, 83858),
            (False, 18020, 114017, -48931, 32606, 21474, 89843),
            (False, -77139, 10506, -89994, -18797, -80, 59318),
            (False, 8476, 79288, -75520, 11602, -96624, -24783),
            (True, -47488, -1262, 24338, 100707, 16292, 72967),
            (False, -84341, 13987, 2429, 92914, -90671, -1318),
            (False, -37810, 49457, -71013, -7894, -105357, -13188),
            (False, -27365, 46395, 31009, 98017, 15428, 76570),
            (False, -70369, -16548, 22648, 78696, -1892, 86821),
            (True, -53470, 21291, -120233, -33476, -44150, 38147),
            (False, -93533, -4276, -16170, 68771, -104985, -24507)
        ], 2758514936282235),
        ("aoc example 1 from part 1", [
            (True, 10, 12, 10, 12, 10, 12),
            (True, 11, 13, 11, 13, 11, 13),
            (False, 9, 11, 9, 11, 9, 11),
            (True, 10, 10, 10, 10, 10, 10),
        ], 39),
        ("aoc own example 1 on only", [
            (True, 10, 12, 10, 12, 10, 12),
            (True, 11, 13, 11, 13, 11, 13),
            (True, 10, 10, 10, 10, 10, 10),
        ], 46),
        ("aoc own example 2 off only", [
            (False, 10, 12, 10, 12, 10, 12),
            (False, 11, 13, 11, 13, 11, 13),
            (False, 10, 10, 10, 10, 10, 10),
        ], 0),
        ("own example 3 big off at the end", [
            (True, -5, 47, -31, 22, -19, 33),
            (True, -44, 5, -27, 21, -14, 35),
            (True, -49, -1, -11, 42, -10, 38),
            (True, -20, 34, -40, 6, -44, 1),
            (False, 26, 39, 40, 50, -2, 11),
            (True, -41, 5, -41, 6, -36, 8),
            (False, -43, -33, -45, -28, 7, 25),
            (True, -33, 15, -32, 19, -34, 11),
            (False, 35, 47, -46, -34, -11, 5),
            (True, -14, 36, -6, 44, -16, 29),
            (True, -57795, -6158, 29564, 72030, 20435, 90618),
            (True, 36731, 105352, -21140, 28532, 16094, 90401),
            (True, 30999, 107136, -53464, 15513, 8553, 71215),
            (True, 13528, 83982, -99403, -27377, -24141, 23996),
            (True, -72682, -12347, 18159, 111354, 7391, 80950),
            (True, -1060, 80757, -65301, -20884, -103788, -16709),
            (True, -83015, -9461, -72160, -8347, -81239, -26856),
            (True, -52752, 22273, -49450, 9096, 54442, 119054),
            (True, -29982, 40483, -108474, -28371, -24328, 38471),
            (True, -4958, 62750, 40422, 118853, -7672, 65583),
            (True, 55694, 108686, -43367, 46958, -26781, 48729),
            (True, -98497, -18186, -63569, 3412, 1232, 88485),
            (True, -726, 56291, -62629, 13224, 18033, 85226),
            (True, -110886, -34664, -81338, -8658, 8914, 63723),
            (True, -55829, 24974, -16897, 54165, -121762, -28058),
            (True, -65152, -11147, 22489, 91432, -58782, 1780),
            (True, -120100, -32970, -46592, 27473, -11695, 61039),
            (True, -18631, 37533, -124565, -50804, -35667, 28308),
            (True, -57817, 18248, 49321, 117703, 5745, 55881),
            (True, 14781, 98692, -1341, 70827, 15753, 70151),
            (True, -34419, 55919, -19626, 40991, 39015, 114138),
            (True, -60785, 11593, -56135, 2999, -95368, -26915),
            (True, -32178, 58085, 17647, 101866, -91405, -8878),
            (True, -53655, 12091, 50097, 105568, -75335, -4862),
            (True, -111166, -40997, -71714, 2688, 5609, 50954),
            (True, -16602, 70118, -98693, -44401, 5197, 76897),
            (True, 16383, 101554, 4615, 83635, -44907, 18747),
            (False, -95822, -15171, -19987, 48940, 10804, 104439),
            (True, -89813, -14614, 16069, 88491, -3297, 45228),
            (True, 41075, 99376, -20427, 49978, -52012, 13762),
            (True, -21330, 50085, -17944, 62733, -112280, -30197),
            (True, -16478, 35915, 36008, 118594, -7885, 47086),
            (False, -98156, -27851, -49952, 43171, -99005, -8456),
            (False, 2032, 69770, -71013, 4824, 7471, 94418),
            (True, 43670, 120875, -42068, 12382, -24787, 38892),
            (False, 37514, 111226, -45862, 25743, -16714, 54663),
            (False, 25699, 97951, -30668, 59918, -15349, 69697),
            (False, -44271, 17935, -9516, 60759, 49131, 112598),
            (True, -61695, -5813, 40978, 94975, 8655, 80240),
            (False, -101086, -9439, -7088, 67543, 33935, 83858),
            (False, 18020, 114017, -48931, 32606, 21474, 89843),
            (False, -77139, 10506, -89994, -18797, -80, 59318),
            (False, 8476, 79288, -75520, 11602, -96624, -24783),
            (True, -47488, -1262, 24338, 100707, 16292, 72967),
            (False, -84341, 13987, 2429, 92914, -90671, -1318),
            (False, -37810, 49457, -71013, -7894, -105357, -13188),
            (False, -27365, 46395, 31009, 98017, 15428, 76570),
            (False, -70369, -16548, 22648, 78696, -1892, 86821),
            (True, -53470, 21291, -120233, -33476, -44150, 38147),
            (False, -93533, -4276, -16170, 68771, -104985, -24507),
            (False, -9999999999, 9999999999, -9999999999, 9999999999, -9999999999, 9999999999),
        ], 0),
        ("aoc own example 4 ", [
            (True, 10, 10, 10, 10, 10, 10),
        ], 1),
        ("aoc own example 5 ", [
            (True, 10, 10, 10, 10, 10, 10),
            (True, 11, 11, 11, 11, 11, 11),
        ], 2),
        ("aoc own example 6 ", [
            (True, 10, 10, 10, 10, 10, 10),
            (True, 10, 10, 11, 11, 10, 10),
            (True, 11, 11, 11, 11, 11, 11),
            (True, 11, 11, 10, 10, 10, 10),
        ], 4),
        ("aoc own example 7 ", [
            (True, 10, 10, 10, 10, 10, 10),
            (True, 10, 10, 11, 11, 10, 10),
            (True, 11, 11, 11, 11, 11, 11),
            (False, 11, 11, 10, 10, 10, 10),
            (True, 11, 11, 10, 10, 10, 10),
        ], 4),
        ("aoc own example 8 ", [
            (True, 10, 10, 10, 10, 10, 10),
            (True, 10, 10, 11, 11, 10, 10),
            (True, 11, 11, 11, 11, 11, 11),
            (True, 11, 11, 10, 10, 10, 10),
            (False, 11, 11, 10, 10, 10, 10),
        ], 3),
        ("aoc own example 9 ", [
            (True, 10, 10, 10, 10, 10, 10),
            (False, 10, 10, 10, 10, 10, 10),
            (True, 10, 10, 11, 11, 10, 10),
            (False, 10, 10, 11, 11, 10, 10),
            (True, 11, 11, 11, 11, 11, 11),
            (False, 11, 11, 11, 11, 11, 11),
            (True, 11, 11, 10, 10, 10, 10),
            (False, 11, 11, 10, 10, 10, 10),
        ], 0),
        ("aoc own example 10 ", [
            (True, 0, 2, 0, 2, 0, 2),
            (False, 1, 1, 1, 1, 1, 1),
        ], 26),
        ("aoc example 2 from part 1", [
            (True, -20, 26, -36, 17, -47, 7),
            (True, -20, 33, -21, 23, -26, 28),
            (True, -22, 28, -29, 23, -38, 16),
            (True, -46, 7, -6, 46, -50, -1),
            (True, -49, 1, -3, 46, -24, 28),
            (True, 2, 47, -22, 22, -23, 27),
            (True, -27, 23, -28, 26, -21, 29),
            (True, -39, 5, -6, 47, -3, 44),
            (True, -30, 21, -8, 43, -13, 34),
            (True, -22, 26, -27, 20, -29, 19),
            (False, -48, -32, 26, 41, -47, -37),
            (True, -12, 35, 6, 50, -50, -2),
            (False, -48, -32, -32, -16, -15, -5),
            (True, -18, 26, -33, 15, -7, 46),
            (False, -40, -22, -38, -28, 23, 41),
            (True, -16, 35, -41, 10, -47, 6),
            (False, -32, -23, 11, 30, -14, 3),
            (True, -49, -5, -3, 45, -29, 18),
            (False, 18, 30, -20, -8, -3, 13),
            (True, -41, 9, -7, 43, -33, 15)
        ], 590784),
        ("aoc own example 11 ", [
            (True, 0, 2, 0, 2, 0, 2),
            (True, 1, 1, 1, 1, 1, 1),
        ], 27),
        ("aoc own example 12 ", [
            (True, 0, 2, 0, 2, 0, 2),
            (True, 0, 2, 0, 2, 0, 2),
            (True, 0, 2, 0, 2, 0, 2),
            (True, 0, 2, 0, 2, 0, 2),
            (True, 0, 2, 0, 2, 0, 2),
            (True, 0, 2, 0, 2, 0, 2),
            (True, 0, 2, 0, 2, 0, 2),
            (True, 0, 2, 0, 2, 0, 2),
            (True, 0, 2, 0, 2, 0, 2),
            (True, 0, 2, 0, 2, 0, 2),
            (True, 0, 2, 0, 2, 0, 2),
            (True, 0, 2, 0, 2, 0, 2),
            (True, 0, 2, 0, 2, 0, 2),
            (True, 0, 2, 0, 2, 0, 2),
        ], 27),
        ("aoc own example 13 ", [
            (True, 0, 2, 0, 2, 0, 2),
            (True, 0, 20, 0, 20, 0, 20),
            (True, 0, 2, 0, 2, 0, 2),
        ], 9261),
        ("aoc own example 14 ", [
            (True, 0, 2, 0, 2, 0, 2),
            (True, 5, 7, 5, 7, 5, 7),
            (True, 0, 7, 0, 7, 0, 7),
        ], 512),
        ("aoc own example 15 ", [
            (True, 0, 2, 0, 2, 0, 2),
            (False, 1000, 2000, 100, 200, 1000, 2000),
            (True, 1000, 2001, 100, 200, 1000, 2000),
            (False, 1000, 2001, 100, 200, 1000, 2000),
        ], 27),
    ])
    def test_count_all_cubes(self, _, data, expected):
        handler = solutions.ReactorHandler(data)
        self.assertEqual(expected, handler.count_all_cubes())

    @parameterized.expand([
        ("own example 1", [
            (True, -2, 3, -2, 2, -2, 2),
            (True, -2, 2, -2, 2, -3, 1),
            (True, -2, 1, -1, 3, -3, -1),
        ]),
        ("own example 2", [
            (True, -20, 26, -36, 17, -47, 7),
            (True, -20, 33, -21, 23, -26, 28),
            (True, -22, 28, -29, 23, -38, 16),
            (True, -46, 7, -6, 46, -50, -1),
            (True, -49, 1, -3, 46, -24, 28),
            (True, 2, 47, -22, 22, -23, 27),
            (True, -27, 23, -28, 26, -21, 29),
            (True, -39, 5, -6, 47, -3, 44),
            (True, -30, 21, -8, 43, -13, 34),
            (True, -22, 26, -27, 20, -29, 19),
            (False, -48, -32, 26, 41, -47, -37),
            (True, -12, 35, 6, 50, -50, -2),
            (False, -48, -32, -32, -16, -15, -5),
            (True, -18, 26, -33, 15, -7, 46),
            (False, -40, -22, -38, -28, 23, 41),
            (True, -16, 35, -41, 10, -47, 6),
            (False, -32, -23, 11, 30, -14, 3),
            (True, -49, -5, -3, 45, -29, 18),
            (False, 18, 30, -20, -8, -3, 13),
            (True, -41, 9, -7, 43, -33, 15)
        ]),
    ])
    def test_compare_part_1_and_part_2_algs(self, _, data):
        handler = solutions.ReactorHandler(data)
        self.assertEqual(handler.count_cubes_in_initialization_area(), handler.count_all_cubes())


# class Day23(unittest.TestCase):
#     @parameterized.expand([
#         ("aoc example 1", [
#             [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
#             [9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9],
#             [9, 9, 9, 2, 9, 3, 9, 2, 9, 4, 9, 9, 9],
#             [9, 9, 9, 1, 9, 4, 9, 3, 9, 1, 9, 9, 9],
#             [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
#         ], 12521)
#     ])
#     def test_organize_amphipods(self, _, data, expected):
#         handler = solutions.AmphipodHandler(data, large_rooms=False)
#         self.assertEqual(expected, handler.organize_amphipods())
#
#     @parameterized.expand([
#         ("aoc example 1", [
#             [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
#             [9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9],
#             [9, 9, 9, 2, 9, 3, 9, 2, 9, 4, 9, 9, 9],
#             [9, 9, 9, 4, 9, 3, 9, 2, 9, 1, 9, 9, 9],
#             [9, 9, 9, 4, 9, 2, 9, 1, 9, 3, 9, 9, 9],
#             [9, 9, 9, 1, 9, 4, 9, 3, 9, 1, 9, 9, 9],
#             [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
#         ], 44169)
#     ])
#     def test_organize_amphipods_large(self, _, data, expected):
#         handler = solutions.AmphipodHandler(data, large_rooms=True)
#         self.assertEqual(expected, handler.organize_amphipods())
#
#     @parameterized.expand([
#         ("own example 1", [
#             [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
#             [9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9],
#             [9, 9, 9, 2, 9, 3, 9, 2, 9, 4, 9, 9, 9],
#             [9, 9, 9, 1, 9, 4, 9, 3, 9, 1, 9, 9, 9],
#             [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
#         ], False, False),
#         ("own example 2", [
#             [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
#             [9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9],
#             [9, 9, 9, 1, 9, 2, 9, 3, 9, 4, 9, 9, 9],
#             [9, 9, 9, 1, 9, 2, 9, 3, 9, 4, 9, 9, 9],
#             [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
#         ], False, True),
#         ("own example 3", [
#             [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
#             [9, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 9],
#             [9, 9, 9, 1, 9, 2, 9, 0, 9, 0, 9, 9, 9],
#             [9, 9, 9, 1, 9, 2, 9, 3, 9, 4, 9, 9, 9],
#             [9, 9, 9, 1, 9, 2, 9, 3, 9, 4, 9, 9, 9],
#             [9, 9, 9, 1, 9, 2, 9, 4, 9, 3, 9, 9, 9],
#             [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
#         ], True, False),
#         ("own example 4", [
#             [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
#             [9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9],
#             [9, 9, 9, 1, 9, 2, 9, 3, 9, 4, 9, 9, 9],
#             [9, 9, 9, 1, 9, 2, 9, 3, 9, 4, 9, 9, 9],
#             [9, 9, 9, 1, 9, 2, 9, 3, 9, 4, 9, 9, 9],
#             [9, 9, 9, 1, 9, 2, 9, 3, 9, 4, 9, 9, 9],
#             [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
#         ], True, True),
#         ("own example 5", [
#             [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
#             [9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9],
#             [9, 9, 9, 1, 9, 2, 9, 3, 9, 4, 9, 9, 9],
#             [9, 9, 9, 1, 9, 2, 9, 3, 9, 4, 9, 9, 9],
#             [9, 9, 9, 1, 9, 2, 9, 3, 9, 4, 9, 9, 9],
#             [9, 9, 9, 2, 9, 1, 9, 3, 9, 4, 9, 9, 9],
#             [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
#         ], True, False),
#     ])
#     def test_game_over(self, _, data, large_rooms, expected):
#         handler = solutions.AmphipodHandler(data, large_rooms)
#         self.assertEqual(expected, handler.game_over())
#
#     @parameterized.expand([
#         ("own example 1", [
#             [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
#             [9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9],
#             [9, 9, 9, 2, 9, 3, 9, 2, 9, 4, 9, 9, 9],
#             [9, 9, 9, 1, 9, 4, 9, 3, 9, 1, 9, 9, 9],
#             [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
#         ], False, {
#              ((2, 3), (1, 1)),
#              ((2, 3), (1, 2)),
#              ((2, 3), (1, 4)),
#              ((2, 3), (1, 6)),
#              ((2, 3), (1, 8)),
#              ((2, 3), (1, 10)),
#              ((2, 3), (1, 11)),
#              ((2, 5), (1, 1)),
#              ((2, 5), (1, 2)),
#              ((2, 5), (1, 4)),
#              ((2, 5), (1, 6)),
#              ((2, 5), (1, 8)),
#              ((2, 5), (1, 10)),
#              ((2, 5), (1, 11)),
#              ((2, 7), (1, 1)),
#              ((2, 7), (1, 2)),
#              ((2, 7), (1, 4)),
#              ((2, 7), (1, 6)),
#              ((2, 7), (1, 8)),
#              ((2, 7), (1, 10)),
#              ((2, 7), (1, 11)),
#              ((2, 9), (1, 1)),
#              ((2, 9), (1, 2)),
#              ((2, 9), (1, 4)),
#              ((2, 9), (1, 6)),
#              ((2, 9), (1, 8)),
#              ((2, 9), (1, 10)),
#              ((2, 9), (1, 11)),
#          }),
#         ("own example 2", [
#             [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
#             [9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9],
#             [9, 9, 9, 1, 9, 2, 9, 3, 9, 4, 9, 9, 9],
#             [9, 9, 9, 1, 9, 2, 9, 3, 9, 4, 9, 9, 9],
#             [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
#         ], False, set()),
#         ("own example 3", [
#             [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
#             [9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 9],
#             [9, 9, 9, 1, 9, 2, 9, 4, 9, 0, 9, 9, 9],
#             [9, 9, 9, 1, 9, 2, 9, 3, 9, 4, 9, 9, 9],
#             [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
#         ], False, {
#              ((2, 7), (2, 9)),
#              ((2, 7), (1, 1)),
#              ((2, 7), (1, 2)),
#              ((2, 7), (1, 4)),
#              ((2, 7), (1, 6)),
#              ((2, 7), (1, 8)),
#          }),
#         ("own example 4", [
#             [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
#             [9, 0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0, 9],
#             [9, 9, 9, 0, 9, 2, 9, 3, 9, 0, 9, 9, 9],
#             [9, 9, 9, 1, 9, 2, 9, 3, 9, 4, 9, 9, 9],
#             [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
#         ], False, {
#              ((1, 8), (2, 3)),
#              ((1, 10), (2, 9)),
#          }),
#         ("own example 5", [
#             [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
#             [9, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 9],
#             [9, 9, 9, 2, 9, 0, 9, 3, 9, 4, 9, 9, 9],
#             [9, 9, 9, 1, 9, 4, 9, 3, 9, 1, 9, 9, 9],
#             [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
#         ], False, {
#              ((2, 3), (1, 1)),
#              ((2, 3), (1, 2)),
#              ((3, 5), (1, 6)),
#              ((3, 5), (1, 8)),
#              ((3, 5), (1, 10)),
#              ((3, 5), (1, 11)),
#              ((2, 9), (1, 6)),
#              ((2, 9), (1, 8)),
#              ((2, 9), (1, 10)),
#              ((2, 9), (1, 11)),
#          }),
#         ("own example 6", [
#             [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
#             [9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9],
#             [9, 9, 9, 1, 9, 2, 9, 3, 9, 4, 9, 9, 9],
#             [9, 9, 9, 1, 9, 2, 9, 3, 9, 4, 9, 9, 9],
#             [9, 9, 9, 1, 9, 2, 9, 3, 9, 4, 9, 9, 9],
#             [9, 9, 9, 1, 9, 2, 9, 3, 9, 4, 9, 9, 9],
#             [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
#         ], True,
#          set()
#          ),
#         ("own example 7", [
#             [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
#             [9, 4, 4, 0, 4, 0, 4, 0, 0, 0, 0, 0, 9],
#             [9, 9, 9, 1, 9, 2, 9, 3, 9, 0, 9, 9, 9],
#             [9, 9, 9, 1, 9, 2, 9, 3, 9, 0, 9, 9, 9],
#             [9, 9, 9, 1, 9, 2, 9, 3, 9, 0, 9, 9, 9],
#             [9, 9, 9, 1, 9, 2, 9, 3, 9, 0, 9, 9, 9],
#             [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
#         ], True, {
#              ((1, 6), (5, 9))
#          }),
#         ("own example 8", [
#             [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
#             [9, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 9],
#             [9, 9, 9, 1, 9, 2, 9, 0, 9, 0, 9, 9, 9],
#             [9, 9, 9, 1, 9, 2, 9, 3, 9, 4, 9, 9, 9],
#             [9, 9, 9, 1, 9, 2, 9, 3, 9, 4, 9, 9, 9],
#             [9, 9, 9, 1, 9, 2, 9, 4, 9, 3, 9, 9, 9],
#             [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
#         ], True, {
#              ((3, 7), (1, 2)),
#              ((3, 7), (1, 4)),
#              ((3, 7), (1, 6)),
#              ((3, 7), (1, 8)),
#              ((3, 7), (1, 10)),
#              ((3, 9), (1, 2)),
#              ((3, 9), (1, 4)),
#              ((3, 9), (1, 6)),
#              ((3, 9), (1, 8)),
#              ((3, 9), (1, 10)),
#          }),
#         ("own example 9", [
#             [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
#             [9, 3, 3, 3, 0, 0, 0, 0, 0, 0, 4, 3, 9],
#             [9, 9, 9, 1, 9, 2, 9, 0, 9, 0, 9, 9, 9],
#             [9, 9, 9, 1, 9, 2, 9, 0, 9, 0, 9, 9, 9],
#             [9, 9, 9, 1, 9, 2, 9, 0, 9, 4, 9, 9, 9],
#             [9, 9, 9, 1, 9, 2, 9, 4, 9, 4, 9, 9, 9],
#             [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
#         ], True, {
#              ((1, 10), (3, 9)),
#              ((5, 7), (3, 9)),
#              ((5, 7), (1, 4)),
#              ((5, 7), (1, 6)),
#              ((5, 7), (1, 8)),
#          }),
#         ("own example 10", [
#             [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
#             [9, 2, 3, 3, 0, 0, 0, 0, 0, 0, 4, 3, 9],
#             [9, 9, 9, 1, 9, 2, 9, 0, 9, 0, 9, 9, 9],
#             [9, 9, 9, 1, 9, 2, 9, 0, 9, 0, 9, 9, 9],
#             [9, 9, 9, 1, 9, 2, 9, 0, 9, 4, 9, 9, 9],
#             [9, 9, 9, 1, 9, 3, 9, 4, 9, 4, 9, 9, 9],
#             [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
#         ], True, {
#              ((1, 10), (3, 9)),
#              ((5, 7), (3, 9)),
#              ((5, 7), (1, 4)),
#              ((5, 7), (1, 6)),
#              ((5, 7), (1, 8)),
#              ((2, 5), (1, 4)),
#              ((2, 5), (1, 6)),
#              ((2, 5), (1, 8)),
#          }),
#
#     ])
#     def test_possible_moves(self, _, data, large_rooms, expected):
#         handler = solutions.AmphipodHandler(data, large_rooms)
#         self.assertEqual(expected, handler.all_possible_moves())


class Day25(unittest.TestCase):
    @parameterized.expand([
        ("aoc example 1", [
         "v...>>.vv>",
         ".vv>>.vv..",
         ">>.>v>...v",
         ">>v>>.>.v.",
         "v>v.vv.v..",
         ">.>>..v...",
         ".vv..>.>v.",
         "v.v..>>v.v",
         "....v..v.>"
         ], 58)
    ])
    def test_count_steps(self, _, data, expected):
        board = [list(line) for line in data]
        handler = solutions.CucumberHandler(board)
        self.assertEqual(expected, handler.count_steps())
