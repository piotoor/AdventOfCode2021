from utilities import non_blank_lines


class Pathfinder:
    def __init__(self, data):
        self.data = data
        self.adjacency_dict = {}
        self.data_to_adjacency_dict()

        self.visited = set()
        self.curr_path = []
        self.paths = []
        self.small_nodes_allowed_visits = {}
        self.reset_small_nodes_allowed_visits()

    def data_to_adjacency_dict(self):
        for x, y in self.data:
            if x in self.adjacency_dict:
                self.adjacency_dict[x].add(y)
            else:
                self.adjacency_dict[x] = {y}
            if y in self.adjacency_dict:
                self.adjacency_dict[y].add(x)
            else:
                self.adjacency_dict[y] = {x}

    def dfs(self, node):
        if node == "end":
            self.paths.append(self.curr_path.copy())
        else:
            for n in self.adjacency_dict[node]:
                if n in self.small_nodes_allowed_visits.keys() and self.small_nodes_allowed_visits[n] == 0:
                    continue

                self.curr_path.append(n)
                if n.islower():
                    self.small_nodes_allowed_visits[n] -= 1

                self.dfs(n)

                if n.islower():
                    self.small_nodes_allowed_visits[n] += 1
                self.curr_path.pop()

    def count_paths_visiting_all_small_caves_once(self):
        self.paths = []
        self.reset_small_nodes_allowed_visits()
        self.curr_path = ["start"]
        self.small_nodes_allowed_visits["start"] = 0
        self.dfs("start")

        return len(self.paths)

    def reset_small_nodes_allowed_visits(self):
        self.small_nodes_allowed_visits = {x: 1 for x in set(filter(lambda x: x.islower(), self.adjacency_dict))}

    def count_paths_visiting_all_small_caves_but_one_once(self):
        self.paths = []

        for small in self.small_nodes_allowed_visits:
            self.reset_small_nodes_allowed_visits()
            self.small_nodes_allowed_visits[small] = 2
            self.curr_path = ["start"]
            self.small_nodes_allowed_visits["start"] = 0
            self.dfs("start")

        return len(set(tuple(p) for p in self.paths))


def parse_day12_data():
    with open("day12.txt", "r") as f:
        data = [tuple(line.split("-")) for line in non_blank_lines(f)]

    return data


def day12_a():
    data = parse_day12_data()
    pf = Pathfinder(data)
    print("day12_a = {}".format(pf.count_paths_visiting_all_small_caves_once()))


def day12_b():
    data = parse_day12_data()
    pf = Pathfinder(data)
    print("day12_b = {}".format(pf.count_paths_visiting_all_small_caves_but_one_once()))
