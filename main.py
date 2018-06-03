import matplotlib.pyplot as plt
from random import uniform, seed

if __name__ == '__main__':
    #seed(0)

    x_train = [x + uniform(-1, 1) for x in range(10)]
    y_train = [y for y in range(10)]

    # plt.plot(x_train, y_train, 'o')
    # plt.show()

    w0 = uniform(0, 1)
    w1 = uniform(0, 1)
    lr = 0.01
    plt.plot([w0 + w1 * x for x in range(10)], range(10))
    plt.plot(x_train, y_train, 'ro')
    plt.show()
    for epoch in range(1000):
        for x, y in zip(x_train, y_train):
            pred  = w1 * x + w0
            loss  = pred - y
            dw1   = loss * x
            dw0   = loss * 1
            w0 -= dw0 * lr
            w1 -= dw1 * lr
    print('w0: ', w0)
    print('w1: ', w1)
    plt.plot([w0 + w1 * x for x in range(10)], range(10))
    plt.plot(x_train, y_train, 'ro')
    plt.show()
