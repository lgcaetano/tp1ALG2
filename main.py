import math
import os
from random import randint
# import matplotlib.pyplot as plt


def str_arr_to_num(arr):
    result = []
    for e in arr:
        result.append(float(e))
    return result

def convert_classes(dataset): #função converte classes de uma base de dados em 0 ou 1
    ref = dataset[0][-1]
    return list(map(lambda point: point[:-1] + ([1] if point[-1] == ref else [0]), dataset))

def read_and_parse_dataset(file): #função para tratar dados de arquivo
    file = open(file, "r")
    dataset = []
    for line in file:
        if not line[0] == "@":
            data = line[:-1].split(",")
            dataset.append(str_arr_to_num(data[:-1]) + data[-1:])
    return convert_classes(dataset) 

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


def split_by_median(median, points, index): #função divide lista de números em duas: 
    smaller_or_equal = [] #uma contendo elementos menores ou iguais à mediana
    greater = [] #uma contendo os elementos maiores que a mediana
    for i in points:
        if i[index] > median:
            greater.append(i)
        else:
            smaller_or_equal.append(i)

    if len(smaller_or_equal) == 0 or len(greater) == 0: #aqui a função verifica se uma das listas está vazia, o que ocorre quando os pontos são iguais 
        return split_middle(points, index) #como isso acaba levando à chamadas infinitas da função (já que a lista não se divide), dividimos a lista ao meio neste caso
    
    return [smaller_or_equal, greater]

def euclid_distance(a, b):
    sum_of_squares = 0
    for i in range(len(a) - 1):
        sum_of_squares += (b[i] - a[i]) ** 2
    return math.sqrt(sum_of_squares)

def treat(num): 
    return num if num else 1

def treat_test_data(data): #função usada para converter números de decimal para porcentagem
    return round(data * 100, 2)


class PriorityQueue:
    def __init__(self, k, ref):
        self.queue = []
        self.k = k
        self.ref_point = ref

    def insert(self, element):
        self.queue.append(element)
        self.queue.sort(key = lambda x: euclid_distance(x, self.ref_point)) # ordena-se a fila para que os elementos mais distantes
        self.queue = self.queue[:self.k] # aqui são mantidos apenas os k primeiros elementos da lista

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
        return 1 if one_count == max_val else 0


class Node:
    def __init__(self, value, depth):
        self.left = None
        self.right = None
        self.value = value
        self.depth = depth # profundidade em que o nó foi inserido na árvore

    def insert_left(self, node):
        self.left = node

    def insert_right(self, node):
        self.right = node

    def is_leaf(self):
        return self.left == None and self.right == None




class KDTree:
    def __build_tree(self, points, depth):

        if len(points) <= 0:
            return None

        if len(points) <= 1:
            return Node([*points[0]], depth) #caso a lista de pontos possui apenas um ponto, retornamos um nó com uma cópia deste ponto armazenada 

        node = None
        index = depth % (len(points[0]) - 1) # index nos indica qual coordenada dos pontos deve ser analisada, já que esta varia de acordo com a profundidade atual
        median = get_median(points, index)
        [p1, p2] = split_by_median(median, points, index)
        node = Node(median, depth) # criamos um nó com o valor da mediana que divide a árvore
        node.insert_left(self.__build_tree(p1, depth + 1))
        node.insert_right(self.__build_tree(p2, depth + 1)) 
        return node

    def __init__(self, points):
        self.head = self.__build_tree(points, 0)

    def __get_leafs(self, node): # método retorna lista com todos os pontos armazenados nos nós folha da árvore 
        if(node == None):
            return []
        if(node.is_leaf()):
            return [node.value]
        else:
            return self.__get_leafs(node.left) + self.__get_leafs(node.right)

    def recursive_find_neighbours(self, node, queue): # método utilizado para percorrer a árvore tentando encontrar 
                                                      # os k vizinhos mais próximos do ponto de referência da fila
        if node == None:
            return

        
        if(node.is_leaf()): # caso um nó folha, que armazena um ponto, for encontrado, inserimos seu ponto na fila
            queue.insert(node.value) # a fila será responsável por avaliar se o ponto é um dos k vizinhos mais próximos 
            return 
        
        median = node.value
        ref = queue.ref_point
        depth = node.depth
        index = depth % (len(ref) - 1) # aqui obtemos o índice da coordenada do ponto que deve ser comparada com a mediana
        
        greater = True if ref[index] > median else False
        first = node.right if greater else node.left # percorremos primeiro a sub árvore em que seria inserido o ponto de ref.
        second = node.left if greater else node.right 
        
        self.recursive_find_neighbours(first, queue)
        
        if queue.is_full() and abs(ref[index] - median) > queue.largest_dist(): #aqui verificamos se a sub-árvore remanescente pode conter um dos k vizinhos mais próximos 
            return  
        
        self.recursive_find_neighbours(second, queue)

    def find_neighbours(self, queue):
        self.recursive_find_neighbours(self.head, queue)
        
class XNN:

    def __init__(self, training_data, testing_data, k):
        self.k = k
        self.test_data = testing_data
        self.kd_tree = KDTree(training_data)
        self.test()

    def predict_class(self, point):
        queue = PriorityQueue(self.k, point)
        self.kd_tree.find_neighbours(queue)
        return queue.most_common() # a previsão é dada pela moda encontrada dentre as classificações dos k vizinhos mais próximos

    def test(self):
        results = [[0, 0],[0, 0]]

        total = len(self.test_data)
        
        for point in self.test_data:
            prediction = self.predict_class(point)
            real_class = point[-1]
            results[prediction][real_class] += 1
        
        real_negatives = results[0][0]
        real_positives = results[1][1]
        fake_negatives = results[0][1]
        fake_positives = results[1][0]

        self.precision = real_positives / treat(real_positives + fake_positives)
        self.recall = real_positives / treat(real_positives + fake_negatives)
        self.accuracy = (real_positives + real_negatives) / treat(total)

    def print_test_results(self):
        print(f"Dados do algoritmo:\nPrecisão: {treat_test_data(self.precision)}%\nRevocação: {treat_test_data(self.recall)}%\nAcurácia: {treat_test_data(self.accuracy)}%\n")

    def get_test_results(self):
        return [self.precision, self.recall, self.accuracy]

# A partir daqui, são implementadas funções que auxiliam no processamento
# dos dados de entrada e na visualização  dos dados de saída



def treat_for_graph(arr): 
    result = []
    for e in arr:
        result.append(treat_test_data(e))
    return result



def split_n_from(arr, n): # essa função divide as bases de dados aleatoriamente de acordo com n passado como parâmetro
    selected = []
    copy = [*arr]
    for i in range(n):
        index = randint(0, len(copy) - 1)
        selected.append(arr[index])
        del copy[index]
    return [selected, copy]

def sum_lists(a, b):
    if len(a) < len(b):
        return [*b]
    result = []
    for i in range(len(a)):
        result.append(a[i] + b[i])
    return result

def divide_by_n(arr, n):
    for i in range(len(arr)):
        arr[i] /= n

def run_algorithm(k):
    files = os.listdir("datasets/") # todas as bases de dados contidas na pasta datasets localizadas no mesmo diretório deste arquivo serão testadas
    summed_data = []

    for file in files:
        data = read_and_parse_dataset(f"datasets/{file}")
        [train_data, test_data] = split_n_from(data, math.floor(len(data) * 0.7)) # dividindo bases de dados na proporção 70/30
        summed_data = sum_lists(summed_data, XNN(train_data, test_data, k).get_test_results())

    divide_by_n(summed_data, len(files))
    print(f"Média das bases:\nPrecisão: {treat_test_data(summed_data[0])}%\nRevocação: {treat_test_data(summed_data[1])}%\nAcurácia: {treat_test_data(summed_data[2])}%")

run_algorithm(1)


