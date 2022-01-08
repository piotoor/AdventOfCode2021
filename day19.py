from utilities import non_blank_lines
import copy


class BeaconHandler:
    def __init__(self, data):
        self.scanners = data
        self.good_scanners = {0: self.scanners[0]}
        self.scanners_positions = {0: (0, 0, 0)}

    @classmethod
    def x1(cls, point):
        x, y, z = point
        return [x, z, -y]

    @classmethod
    def x2(cls, point):
        x, y, z = point
        return [x, -y, -z]

    @classmethod
    def x3(cls, point):
        x, y, z = point
        return [x, -z, y]

    @classmethod
    def y1(cls, point):
        x, y, z = point
        return [-z, y, x]

    @classmethod
    def y2(cls, point):
        x, y, z = point
        return [-x, y, -z]

    @classmethod
    def y3(cls, point):
        x, y, z = point
        return [z, y, -x]

    @classmethod
    def z1(cls, point):
        x, y, z = point
        return [y, -x, z]

    @classmethod
    def z2(cls, point):
        x, y, z = point
        return [-x, -y, z]

    @classmethod
    def z3(cls, point):
        x, y, z = point
        return [-y, x, z]

    @classmethod
    def rotate(cls, point, rot):
        for r in rot:
            if r > 8 or r < 0:
                continue
            point = BeaconHandler.rotations_table[r](point)
        return point

    @classmethod
    def generate_all_rotations(cls, points):
        first_rot = [-1, 0, 1, 2, 6, 8]
        second_rot = [-1, 3, 4, 5]

        ans = {}

        for f in first_rot:
            for s in second_rot:
                rot = (f, s)
                ans[rot] = []
                for p in points:
                    ans[rot].append(BeaconHandler.rotate(p, rot))

        return ans

    @classmethod
    def calculate_manhattan_distance(cls, point_a, point_b):
        a_x, a_y, a_z = point_a
        b_x, b_y, b_z = point_b

        return sum((abs(a_x - b_x), abs(a_y - b_y), abs(a_z - b_z)))

    def do_scanners_overlap(self, good_ind, bad_ind):
        bad_scanner_all_rotations = BeaconHandler.generate_all_rotations(self.scanners[bad_ind])

        for rot in bad_scanner_all_rotations:
            diffs = {}
            for rotated_point in bad_scanner_all_rotations[rot]:
                for adjusted_point in self.good_scanners[good_ind]:
                    a_x, a_y, a_z = adjusted_point
                    r_x, r_y, r_z = rotated_point
                    diff = (a_x - r_x, a_y - r_y, a_z - r_z)
                    if diff in diffs:
                        diffs[diff] += 1
                    else:
                        diffs[diff] = 1

            most_freq_diff = max(diffs, key=diffs.get)
            max_diff_val = max(diffs.values())
            if max_diff_val >= 12:
                return most_freq_diff, bad_scanner_all_rotations[rot]

        return None, None

    @classmethod
    def adjust_scanner_position(cls, scanner, diff):
        adjusted_scanner = copy.deepcopy(scanner)

        for i in range(len(scanner)):
            adjusted_scanner[i][0] += diff[0]
            adjusted_scanner[i][1] += diff[1]
            adjusted_scanner[i][2] += diff[2]

        return adjusted_scanner

    def compute_scanners_positions(self):
        while len(self.scanners_positions) < len(self.scanners):
            for i in range(len(self.scanners)):
                if i in self.scanners_positions:
                    continue

                for j in self.scanners_positions:
                    diff, rotated_scanner = self.do_scanners_overlap(j, i)
                    if diff is None and rotated_scanner is None:
                        continue

                    self.good_scanners[i] = BeaconHandler.adjust_scanner_position(rotated_scanner, diff)
                    self.scanners_positions[i] = diff
                    break

    def count_beacons(self):
        self.compute_scanners_positions()
        beacons = set()
        for _, s in self.good_scanners.items():
            for b in s:
                beacons.add(tuple(b))

        return len(beacons)

    def find_largest_manhattan_distance(self):
        self.compute_scanners_positions()
        ans = 0

        for a in self.scanners_positions:
            for b in self.scanners_positions:
                if a == b:
                    continue

                pos_a = self.scanners_positions[a]
                pos_b = self.scanners_positions[b]
                manhattan_dist_a_b = BeaconHandler.calculate_manhattan_distance(pos_a, pos_b)
                ans = max(ans, manhattan_dist_a_b)

        return ans


BeaconHandler.rotations_table = [
    BeaconHandler.x1,
    BeaconHandler.x2,
    BeaconHandler.x3,

    BeaconHandler.y1,
    BeaconHandler.y2,
    BeaconHandler.y3,

    BeaconHandler.z1,
    BeaconHandler.z2,
    BeaconHandler.z3,
]


def parse_day19_data():
    with open("day19.txt", "r") as f:
        data = []
        for line in non_blank_lines(f):
            if line[0:3] == '---':
                data.append([])
            else:
                data[-1].append(list(map(int, line.split(","))))

    return data


def day19_a():
    data = parse_day19_data()
    handler = BeaconHandler(data)
    print("day19_a = {}".format(handler.count_beacons()))


def day19_b():
    data = parse_day19_data()
    handler = BeaconHandler(data)
    print("day19_b = {}".format(handler.find_largest_manhattan_distance()))


def day19():
    day19_a()
    day19_b()
