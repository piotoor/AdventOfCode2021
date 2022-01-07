import collections
import copy
from utilities import non_blank_lines


class ImageEnhancer:
    def __init__(self, data):
        self.alg, self.img = data

    @classmethod
    def apply_padding(cls, img, padding_char, padding_size):
        img_cols = len(img[0])
        padded_img = []

        for i in range(padding_size):
            padded_img.append([padding_char] * (img_cols + padding_size * 2))
        for row in img:
            padded_img.append([padding_char] * padding_size + row + [padding_char] * padding_size)
        for i in range(padding_size):
            padded_img.append([padding_char] * (img_cols + padding_size * 2))

        return padded_img

    def enhance_image(self, iterations):
        for i in range(iterations):
            target_img = copy.deepcopy(self.img)
            target_img = ImageEnhancer.apply_padding(target_img, '0', 1)

            if i % 2 == 0:
                self.img = ImageEnhancer.apply_padding(self.img, '0', 2)
                target_img = ImageEnhancer.apply_padding(target_img, '1', 1)
            else:
                self.img = ImageEnhancer.apply_padding(self.img, '1', 1)

            for r in range(1, len(target_img) - 1):
                for c in range(1, len(target_img[0]) - 1):
                    binary = ""
                    for rr in range(r - 1, r + 2):
                        for cc in range(c - 1, c + 2):
                            binary += self.img[rr][cc]

                    alg_ind = int(binary, 2)
                    if self.alg[alg_ind] == '#':
                        target_img[r][c] = '1'
                    else:
                        target_img[r][c] = '0'

            self.img = copy.deepcopy(target_img)

    def count_lit_pixels(self, iterations):
        self.enhance_image(iterations)
        c = collections.Counter()
        for row in self.img:
            c.update(row)

        return c['1']


def parse_day20_data():
    with open("day20.txt", "r") as f:
        data = [line for line in non_blank_lines(f)]

    alg = data[0]
    img = [list(map(lambda x: '1' if x == '#' else '0', line)) for line in data[1:]]
    return alg, img


def day20_a():
    data = parse_day20_data()
    enhancer = ImageEnhancer(data)
    print("day20_a = {}".format(enhancer.count_lit_pixels(2)))


def day20_b():
    data = parse_day20_data()
    enhancer = ImageEnhancer(data)
    print("day20_b = {}".format(enhancer.count_lit_pixels(50)))
