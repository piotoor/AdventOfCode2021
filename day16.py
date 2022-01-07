import math


class TransmissionHandler:
    def __init__(self, data):
        self.data_hex = data
        self.data_bin = "".join([format(int(digit, 16), '04b') for digit in self.data_hex])
        self.ind = 0
        self.sum_of_version_numbers = 0

    def parse_packet(self, data):
        ver_int = int(data[self.ind:self.ind+3], 2)
        self.ind += 3
        type_int = int(data[self.ind:self.ind+3], 2)
        self.ind += 3
        self.sum_of_version_numbers += ver_int

        if type_int == 4:
            literal = ""
            while True:
                more = data[self.ind:self.ind+1]
                self.ind += 1
                literal += data[self.ind:self.ind+4]
                self.ind += 4
                if more == '0':
                    break
            return int(literal, 2)
        else:
            length_type_id_str = data[self.ind:self.ind+1]
            self.ind += 1

            if length_type_id_str == '0':
                length_field_size = 15
            else:
                length_field_size = 11

            length_str = data[self.ind:self.ind+length_field_size]
            self.ind += length_field_size
            length_int = int(length_str, 2)

            operands_start_ind = self.ind
            operands = []
            if length_field_size == 15:
                while self.ind < operands_start_ind + length_int:
                    operands.append(self.parse_packet(data))
            else:
                for i in range(length_int):
                    operands.append(self.parse_packet(data))

        return TransmissionHandler.operator(type_int, operands)

    @classmethod
    def operator(cls, type_id, operands):
        if type_id == 0:
            return sum(operands)
        if type_id == 1:
            return math.prod(operands)
        if type_id == 2:
            return min(operands)
        if type_id == 3:
            return max(operands)
        if type_id == 5:
            return 1 if operands[0] > operands[1] else 0
        if type_id == 6:
            return 1 if operands[0] < operands[1] else 0
        if type_id == 7:
            return 1 if operands[0] == operands[1] else 0

    def calculate_sum_of_packet_versions(self):
        self.ind = 0
        self.sum_of_version_numbers = 0
        self.parse_packet(self.data_bin)
        return self.sum_of_version_numbers

    def calculate_value_of_the_outermost_packet(self):
        self.ind = 0
        return self.parse_packet(self.data_bin)


def parse_day16_data():
    with open("day16.txt", "r") as f:
        data = f.readline()
    return data


def day16_a():
    data = parse_day16_data()
    handler = TransmissionHandler(data)
    print("day16_a = {}".format(handler.calculate_sum_of_packet_versions()))


def day16_b():
    data = parse_day16_data()
    handler = TransmissionHandler(data)
    print("day16_b = {}".format(handler.calculate_value_of_the_outermost_packet()))
