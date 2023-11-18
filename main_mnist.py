import matplotlib.pyplot as pyplot
import numpy
from neural_network import neural_network

if __name__ == '__main__':
    input = 784
    hidden = 100
    output = 10
    learing_rate = 0.2
    epochs = 2

    n = neural_network(input, hidden, output, learing_rate)

    data1 = open('/Users/jihunjang/Downloads/mnist_dataset_fullset/mnist_train.csv', 'r')
    train_list = data1.readlines()
    data1.close()

    data2 = open('/Users/jihunjang/Downloads/mnist_dataset_fullset/mnist_test.csv', 'r')
    test_list = data2.readlines()
    data2.close()

    # train
    for e in range(epochs):
        for r in train_list:
            val = r.split(',')
            inputs = (numpy.asfarray(val[1:]) / 255.0 * 0.99) + 0.01
            targets = numpy.zeros(output) + 0.01
            targets[int(val[0])] = 0.99
            n.train(inputs, targets)

    # query
    score = []
    for r in test_list:
        val = r.split(',')
        correct_label = int(val[0])
        inputs = (numpy.asfarray(val[1:]) / 255.0 * 0.99) + 0.01
        outputs = n.query(inputs)
        label = numpy.argmax(outputs)

        if (label == correct_label):
            score.append(1)
        else:
            score.append(0)

    score_arr = numpy.asarray(score)
    print('performance: ', score_arr.sum() / score_arr.size)

