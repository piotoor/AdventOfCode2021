from utilities import non_blank_lines


class AssemblyHandler:
    def __init__(self, program):
        self.program = program
        self.regs = [0, 0, 0, 0]
        self.reg_names = {'w': 0, 'x': 1, 'y': 2, 'z': 3}
        self.subprograms = []
        self.parameters = []

    def dump(self):
        w, x, y, z = self.regs
        print("w = {}\nx = {}\ny = {}\nz = {}\n".format(w, x, y, z))

    def dump_short(self):
        print(self.regs)

    def get_regs(self):
        return self.regs

    def clear_regs(self):
        self.regs = [0, 0, 0, 0]

    def split_program(self):
        self.subprograms.clear()
        for x in self.program:
            if x[0] == "inp":
                self.subprograms.append([])

            self.subprograms[-1].append(x)

    def extract_parameters(self):
        self.parameters.clear()
        for x in self.subprograms:
            a, b, c = 0, 0, 0
            a = int(x[4][1].split()[1])
            b = int(x[5][1].split()[1])
            c = int(x[15][1].split()[1])
            self.parameters.append((a, b, c))

    def print_parameters(self):
        print("   n      a    b    c")
        print("----------------------")
        i = 0
        for x in self.parameters:
            a, b, c = x
            print("{:4} | {:4} {:4} {:4}".format(i, a, b, c))
            i += 1
            if i % 5 == 0:
                print()

    def find_largest_model(self):
        self.split_program()
        self.extract_parameters()
        self.print_parameters()
        w = [0] * 14
        stack = []

        for i in range(len(self.parameters)):
            a, b, c = self.parameters[i]
            if a == 1:          # push
                stack.append((i, c))
                w[i] = 9
            elif a == 26:       # pop
                top_i, top_c = stack[-1]
                w[i] = w[top_i] + top_c + b
                if w[i] > 9:
                    w[top_i] -= w[i] - 9
                    w[i] = 9

                stack.pop()

        w_str = "".join(map(str, w))
        if self.run_program(self.program, w_str)[3] == 0:
            return w_str
        else:
            return None

    def find_smallest_model(self):
        self.split_program()
        self.extract_parameters()
        self.print_parameters()
        w = [0] * 14
        stack = []

        for i in range(len(self.parameters)):
            a, b, c = self.parameters[i]
            if a == 1:          # push
                stack.append((i, c))
                w[i] = 1
            elif a == 26:       # pop
                top_i, top_c = stack[-1]
                w[i] = w[top_i] + top_c + b
                if w[i] < 1:
                    w[top_i] = 1 - (top_c + b)
                    w[i] = 1

                stack.pop()

        w_str = "".join(map(str, w))
        if self.run_program(self.program, w_str)[3] == 0:
            return w_str
        else:
            return None

    def run_program(self, program, cin, clear_regs=True):
        if clear_regs:
            self.clear_regs()
        cin = cin[::-1]
        cin = list(cin)
        for line in program:
            instr, args = line
            args = args.split()
            if len(args) == 1:
                args.append(" ")

            a, b = args

            if instr == "inp":
                self.inp(a, cin[-1])
                cin.pop()
            elif instr == "add":
                if b.isdigit():
                    self.add(a, b)
                else:
                    self.add(a, b)
            elif instr == "mul":
                if b.isdigit():
                    self.mul(a, b)
                else:
                    self.mul(a, b)
            elif instr == "div":
                if b.isdigit():
                    self.div(a, b)
                else:
                    self.div(a, b)
            elif instr == "mod":
                if b.isdigit():
                    self.mod(a, b)
                else:
                    self.mod(a, b)
            elif instr == "eql":
                if b.isdigit():
                    self.eql(a, b)
                else:
                    self.eql(a, b)

        # self.dump()
        return self.get_regs()

    @classmethod
    def is_digit(cls, n):
        try:
            int(n)
            return True
        except ValueError:
            return False

    def inp(self, a, b):
        self.regs[self.reg_names[a]] = int(b)

    def add(self, a, b):
        bb = int(b) if AssemblyHandler.is_digit(b) else self.regs[self.reg_names[b]]
        self.regs[self.reg_names[a]] += bb

    def mul(self, a, b):
        bb = int(b) if AssemblyHandler.is_digit(b) else self.regs[self.reg_names[b]]
        self.regs[self.reg_names[a]] *= bb

    def div(self, a, b):
        bb = int(b) if AssemblyHandler.is_digit(b) else self.regs[self.reg_names[b]]
        self.regs[self.reg_names[a]] //= bb

    def mod(self, a, b):
        bb = int(b) if AssemblyHandler.is_digit(b) else self.regs[self.reg_names[b]]
        self.regs[self.reg_names[a]] %= bb

    def eql(self, a, b):
        bb = int(b) if AssemblyHandler.is_digit(b) else self.regs[self.reg_names[b]]
        self.regs[self.reg_names[a]] = 1 if self.regs[self.reg_names[a]] == bb else 0


def parse_day24_data():
    with open("day24.txt", "r") as f:
        return [(line.split()[0], " ".join(line.split()[1:])) for line in non_blank_lines(f)]


def day24_a():
    data = parse_day24_data()
    handler = AssemblyHandler(data)
    print("day24_a = {}".format(handler.find_largest_model()))


def day24_b():
    data = parse_day24_data()
    handler = AssemblyHandler(data)
    print("day24_b = {}".format(handler.find_smallest_model()))
