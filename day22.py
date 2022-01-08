from utilities import non_blank_lines
import re


class ReactorHandler:
    def __init__(self, data):
        self.data = data

    @classmethod
    def cuboids_overlap(cls, a, b):
        a_minx, a_maxx, a_miny, a_maxy, a_minz, a_maxz = a
        b_minx, b_maxx, b_miny, b_maxy, b_minz, b_maxz = b

        if b_minx > a_maxx or a_minx > b_maxx or b_miny > a_maxy or a_miny > b_maxy or b_minz > a_maxz or a_minz > b_maxz:
            return True
        else:
            return False

    @classmethod
    def cuboid_diff(cls, a, b):
        ans = []
        a_minx, a_maxx, a_miny, a_maxy, a_minz, a_maxz = a
        b_minx, b_maxx, b_miny, b_maxy, b_minz, b_maxz = b

        if ReactorHandler.cuboids_overlap(a, b):
            return [a]

        if a_minx < b_minx:  # L exists
            l_minx = min(a_minx, b_minx)
            l_maxx = max(a_minx, b_minx)
            l_miny = a_miny
            l_maxy = a_maxy
            l_minz = a_minz
            l_maxz = a_maxz

            ans.append((l_minx, l_maxx, l_miny, l_maxy, l_minz, l_maxz))
            if ans[-1] == a:
                return ans

        if a_maxx > b_maxx:  # R exists
            r_minx = min(a_maxx, b_maxx)
            r_maxx = max(a_maxx, b_maxx)
            r_miny = a_miny
            r_maxy = a_maxy
            r_minz = a_minz
            r_maxz = a_maxz

            ans.append((r_minx, r_maxx, r_miny, r_maxy, r_minz, r_maxz))
            if ans[-1] == a:
                return ans

        if a_maxz > b_maxz:  # F exists
            f_minx = max(a_minx, b_minx)
            f_maxx = min(a_maxx, b_maxx)
            f_miny = a_miny
            f_maxy = a_maxy
            f_minz = min(a_maxz, b_maxz)
            f_maxz = max(a_maxz, b_maxz)

            ans.append((f_minx, f_maxx, f_miny, f_maxy, f_minz, f_maxz))
            if ans[-1] == a:
                return ans

        if a_minz < b_minz:  # B exists
            bb_minx = max(a_minx, b_minx)
            bb_maxx = min(a_maxx, b_maxx)
            bb_miny = a_miny
            bb_maxy = a_maxy
            bb_minz = min(a_minz, b_minz)
            bb_maxz = max(a_minz, b_minz)

            ans.append((bb_minx, bb_maxx, bb_miny, bb_maxy, bb_minz, bb_maxz))
            if ans[-1] == a:
                return ans

        if a_miny < b_miny:  # D exists
            d_minx = max(a_minx, b_minx)
            d_maxx = min(a_maxx, b_maxx)
            d_miny = min(a_miny, b_miny)
            d_maxy = max(a_miny, b_miny)
            d_minz = max(a_minz, b_minz)
            d_maxz = min(a_maxz, b_maxz)

            ans.append((d_minx, d_maxx, d_miny, d_maxy, d_minz, d_maxz))
            if ans[-1] == a:
                return ans

        if a_maxy > b_maxy:  # U exists
            u_minx = max(a_minx, b_minx)
            u_maxx = min(a_maxx, b_maxx)
            u_miny = min(a_maxy, b_maxy)
            u_maxy = max(a_maxy, b_maxy)
            u_minz = max(a_minz, b_minz)
            u_maxz = min(a_maxz, b_maxz)

            ans.append((u_minx, u_maxx, u_miny, u_maxy, u_minz, u_maxz))
            if ans[-1] == a:
                return ans

        return ans

    @classmethod
    def cuboid_volume(cls, cuboid):
        x_min, x_max, y_min, y_max, z_min, z_max = cuboid
        return abs((x_max - x_min) * (y_max - y_min) * (z_max - z_min))

    def count_cubes_in_initialization_area(self):
        area_x_min = area_y_min = area_z_min = -50
        area_x_max = area_y_max = area_z_max = 51

        area = set()

        for d in self.data:
            on, x_min, x_max, y_min, y_max, z_min, z_max = d
            for x in range(max(x_min, area_x_min), min(x_max + 1, area_x_max)):
                for y in range(max(y_min, area_y_min), min(y_max + 1, area_y_max)):
                    for z in range(max(z_min, area_z_min), min(z_max + 1, area_z_max)):
                        if on:
                            area.add((x, y, z))
                        else:
                            area.discard((x, y, z))

        return len(area)

    def convert_data(self, data):
        ans = []

        for x in data:
            on, x0, x1, y0, y1, z0, z1 = x
            ans.append((on, x0, x1 + 1, y0, y1 + 1, z0, z1 + 1))

        return ans

    def count_all_cubes(self):
        area = set()

        converted_data = self.convert_data(self.data)

        i = 0
        for x in converted_data:
            i += 1
            on = x[0]
            cuboid = x[1:]
            tmp_area = set()
            for c in area:
                new_cuboids = ReactorHandler.cuboid_diff(c, cuboid)

                for nc in new_cuboids:
                    tmp_area.add(nc)

            if on:
                tmp_area.add(cuboid)

            area.clear()
            area = tmp_area.copy()

        ans = 0

        for x in area:
            ans += ReactorHandler.cuboid_volume(x)
        return ans


def parse_day22_data():
    data = []
    with open("day22.txt", "r") as f:
        for line in non_blank_lines(f):
            tmp = line.split(" ")
            data.append(tuple([tmp[0] == "on"]) + tuple(map(int, re.findall(r"-?\d+", tmp[1]))))
    return data


def day22_a():
    data = parse_day22_data()
    handler = ReactorHandler(data)
    print("day22_a = {}".format(handler.count_cubes_in_initialization_area()))


def day22_b():
    data = parse_day22_data()
    handler = ReactorHandler(data)
    print("day22_b = {}".format(handler.count_all_cubes()))


def day22():
    day22_a()
    day22_b()
