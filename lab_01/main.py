from perceptron1.perceptron import Perceptron
from perceptron1.layers import *
from prettytable import PrettyTable
from training_dataset import dataset, test_dataset, NUMBER_COUNT
import random

# 2**11 = 2048
# 56

def train_perceptron():
    network = Perceptron()
    input_count = len(dataset[0].inputs)
    generating_table = PrettyTable()

    generating_table.field_names = ['Генерация слоев']
    
    for _ in range(input_count):
        network.s_layer.add_neuron(None, lambda value: value)

    generating_table.add_row(['S-слой сгенерирован'])

    a_neurons_count = 2 ** 11
    for position in range(a_neurons_count):
        neuron = ANeuron(None, lambda value: int(value >= 0))
        neuron.input_weights = [
            random.choice([-1, 0, 1]) for i in range(input_count)
        ]
        neuron.calculate_bias()
        network.a_layer.neurons.append(neuron)
    generating_table.add_row(['A-слой сгенерирован'])
    for _ in range(NUMBER_COUNT):
        network.r_layer.add_neuron(a_neurons_count, lambda: 0, lambda value: 1 if value >=0 else -1, 0.01, 0)
        
    generating_table.add_row(['R-слой сгенерирован'])

    print(generating_table)
    
    network.train(dataset)
    network.optimize(dataset)
    return network


def test_network(network):
    total_classifications = len(test_dataset) 
    misc = 0
    for data in test_dataset:
        results = network.solve(data.inputs)
        if results != data.results:
            misc += 1

    accuracy_table = PrettyTable()

    accuracy_table.field_names = ['Точность на тестовых данных']
    accuracy_table.add_row(['{:.2f}%'.format(
            float(total_classifications - misc) / total_classifications * 100
        )])

    print(accuracy_table)


def main():
    network = train_perceptron()
    test_network(network)


if __name__ == '__main__':
    main()
