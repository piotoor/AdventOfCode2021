from utilities import non_blank_lines


class CucumberHandler:
    def __init__(self, data):
        self.board = data

    def count_steps(self):
        moved = True
        step_count = 0
        rr = len(self.board)
        cc = len(self.board[0])

        while moved:
            step_count += 1
            moved = False
            r = 0
            while r < rr:
                c = 0
                first = self.board[r][0]
                last = self.board[r][-1]
                while c < cc:
                    if self.board[r][c] == '>':
                        if self.board[r][(c + 1) % cc] == '.':
                            # print("{} -> {}".format((r, c), (r, (c + 1) % cc)))
                            self.board[r][c], self.board[r][(c + 1) % cc] = self.board[r][(c + 1) % cc], self.board[r][c]
                            moved = True
                            c += 1
                    c += 1
                if first == '>' and last == '>' and self.board[r][-1] == '.':
                    self.board[r][0], self.board[r][-1] = self.board[r][-1], self.board[r][0]
                r += 1

            c = 0
            while c < cc:
                r = 0
                first = self.board[0][c]
                last = self.board[-1][c]
                while r < rr:
                    if self.board[r][c] == 'v':
                        if self.board[(r + 1) % rr][c] == '.':
                            self.board[r][c], self.board[(r + 1) % rr][c] = self.board[(r + 1) % rr][c], self.board[r][c]
                            moved = True
                            r += 1
                    r += 1
                if first == 'v' and last == 'v' and self.board[-1][c] == '.':
                    self.board[0][c], self.board[-1][c] = self.board[-1][c], self.board[0][c]
                c += 1

        return step_count


def parse_day25_data():
    with open("day25.txt", "r") as f:
        data = [list(line) for line in non_blank_lines(f)]
    return data


def day25_a():
    data = parse_day25_data()
    handler = CucumberHandler(data)
    print("day25_a = {}".format(handler.count_steps()))


def day25():
    day25_a()
