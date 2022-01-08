import math
import os
from random import randint


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
    

def str_arr_to_num(arr):
    result = []
    for e in arr:
        result.append(float(e))
    return result

def convert_classes(dataset):
    ref = dataset[0][-1]
    return list(map(lambda point: point[:-1] + ([1] if point[-1] == ref else [0]), dataset))

def read_and_parse_dataset(file):
    file = open(file, "r")
    dataset = []
    for line in file:
        if not line[0] == "@":
            data = line[:-1].split(",")
            dataset.append(str_arr_to_num(data[:2]) + data[-1:])
    return convert_classes(dataset)

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

def euclid_distance(a, b):
    return math.sqrt(((b[0] - a[0]) ** 2) + ((b[1] - a[1]) ** 2))


class PriorityQueue:
    def __init__(self, k, ref):
        self.queue = []
        self.k = k
        self.ref_point = ref

    def insert(self, element):
        self.queue.append(element)
        self.queue.sort(key = lambda x: euclid_distance(x, self.ref_point))
        self.queue = self.queue[:self.k]

    def is_full(self):
        return len(self.queue) >= self.k

    def largest_dist(self):
        return euclid_distance(self.ref_point, self.queue[-1])

    def most_common(self):
        zero_count = 0
        one_count = 0
        for point in self.queue:
            zero_count += 1 if not point[-1] else 0
            one_count += 1 if point[-1] else 0
        max_val = max(zero_count, one_count)
        return 1 if zero_count == max_val else 0


class Node:
    def __init__(self, value, depth):
        self.left = None
        self.right = None
        self.value = value
        self.depth = depth

    def insert_left(self, node):
        self.left = node

    def insert_right(self, node):
        self.right = node

    def is_leaf(self):
        return self.left == None and self.right == None




class KDTree:
    def __build_tree(self, points, depth):
        if len(points) <= 1:
            return Node([*points[0]], depth)
        node = None
        index = depth % 2
        median = get_median(points, index)
        [p1, p2] = split_by_median(median, points, index)
        node = Node(median, depth)
        node.insert_left(self.__build_tree(p1, depth + 1))
        node.insert_right(self.__build_tree(p2, depth + 1))
        return node

    def __init__(self, points):
        self.head = self.__build_tree(points, 0)

    def __get_leafs(self, node):
        if(node == None):
            return []
        if(node.is_leaf()):
            return [node.value]
        else:
            return self.__get_leafs(node.left) + self.__get_leafs(node.right)

    def print_leafs(self):
        print(len(self.__get_leafs(self.head)))

    def recursive_find_neighbours(self, node, queue):
        
        median = node.value
        
        if(node.is_leaf()):
            queue.insert(median)
            return 
        
        ref = queue.ref_point
        depth = node.depth
        index = depth % 2
        
        greater = True if ref[index] > median else False
        first = node.right if greater else node.left
        second = node.left if greater else node.right
        
        self.recursive_find_neighbours(first, queue)
        
        if queue.is_full() and abs(ref[index] - median) > queue.largest_dist():
            return
        
        self.recursive_find_neighbours(second, queue)

    def find_neighbours(self, queue):
        self.recursive_find_neighbours(self.head, queue)
        
class XNN:

    def predict_class(self, point):
        queue = PriorityQueue(self.k, point)
        self.kd_tree.find_neighbours(queue)
        return queue.most_common()

    def __init__(self, training_data, testing_data, k):
        self.k = k
        self.test_data = testing_data
        self.kd_tree = KDTree(training_data)
        self.test()

    def test(self):
        matrix = [[0, 0],[0, 0]]

        total = len(self.test_data)
        
        for point in self.test_data:
            prediction = self.predict_class(point)
            real_class = point[-1]
            matrix[prediction][real_class] += 1
        
        real_negatives = matrix[0][0]
        real_positives = matrix[1][1]
        fake_negatives = matrix[0][1]
        fake_positives = matrix[1][0]

        self.precision = real_positives / (real_positives + fake_positives)
        self.recall = real_positives / (real_positives + fake_negatives)
        self.accuracy = (real_positives + real_negatives) / total

    def print_test_results(self):
        print(f"Dados do algoritmo:\nPrecisão: {self.precision}\nRevocação: {self.recall}\nAcurácia:{self.accuracy}")

def split_n_from(arr, n):
    selected = []
    copy = [*arr]
    for i in range(n):
        index = randint(0, len(copy) - 1)
        selected.append(arr[index])
        del copy[index]
    return [selected, copy]

def get_data_from_sets(datasets, limit):
    data = []
    for dataset in datasets:
        data += read_and_parse_dataset(f"datasets/{dataset}")[:limit]
    return data

files = os.listdir("datasets/")

[training_datasets, test_datasets] = split_n_from(files, 3)
training_data = get_data_from_sets(training_datasets, 1000)
test_data = get_data_from_sets(test_datasets, 1000)

model = XNN(training_data, test_data, 1)

model.print_test_results()
