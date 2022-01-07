def equal_points(a, b):
    if a[0] == b[0] and a[1] == b[1]:
        return True
    return False

def contains_point(arr, point):
    for e in arr:
        if equal_points(e, point):
            return True
    return False


def no_duplicates(arr):
    result = []
    for e in arr:
        if not contains_point(result, e):
            result.append(e)
    return result
    
        


def check_equals(arr):
    arr.sort(key = lambda x: x[1])
    arr.sort(key = lambda x: x[0])
    buffer = arr[-1]
    for e in arr:
        if equal_points(buffer, e):
            return True
        else:
            buffer = e
    return False
    

def str_arr_to_int(arr):
    result = []
    for e in arr:
        result.append(int(e))
    return result

def read_and_parse_dataset(file):
    file = open(file, "r")
    dataset = []
    for line in file:
        if not line[0] == "@":
            data = line[:-1].split(",")
            dataset.append(str_arr_to_int(data[:2]) + data[-1:])
    return dataset

def matrix_copy(matrix):
    copy = []
    for i in matrix:
        row = []
        for j in matrix[i]:
            row.append(j)
        copy.append(row)
    return copy

def get_median(points, index):
    copy = [*points]
    copy.sort(key = lambda x: x[index])
    if len(copy) % 2:
        return copy[len(copy) // 2][index]
    else:
        return (copy[len(copy) // 2][index] + copy[(len(copy) // 2) - 1][index]) / 2

def split_middle(points, index):
    points.sort(key = lambda x: x[index])
    middle = (len(points) + 1) // 2
    return[points[:middle], points[middle:]]


def split_by_median(median, points, index):
    smaller_or_equal = []
    greater = []
    for i in points:
        if i[index] > median:
            greater.append(i)
        else:
            smaller_or_equal.append(i)

    if len(smaller_or_equal) == 0 or len(greater) == 0:
        return split_middle(points, index)
    
    return [smaller_or_equal, greater]


class Node:
    def __init__(self, value):
        self.left = None
        self.right = None
        self.value = value

    def insert_left(self, node):
        self.left = node

    def insert_right(self, node):
        self.right = node


class KDTree:
    def __build_tree(self, points, depth):
        if len(points) <= 1:
            # print(points[0])
            return Node([*points[0]])
        node = None
        if depth % 2:
            index = 1
        else:
            index = 0
        median = get_median(points, index)
        # print(len(points))
        [p1, p2] = split_by_median(median, points, index)
        node = Node(median)
        node.insert_left(self.__build_tree(p1, depth + 1))
        node.insert_right(self.__build_tree(p2, depth + 1))
        return node

    def __init__(self, points):
        self.head = self.__build_tree(points, 0)

    def __get_leafs(self, node):
        if(node == None):
            return []
        if(node.left == None and node.right == None):
            return [node.value]
        else:
            return self.__get_leafs(node.left) + self.__get_leafs(node.right)

    def print_leafs(self):
        print(self.__get_leafs(self.head))

points = read_and_parse_dataset("datasets/pima.dat")

tree = KDTree(points)
tree.print_leafs()
# tree.print_leafs()
