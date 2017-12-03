from sklearn import datasets as ds
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import random
import matplotlib.pyplot as plt



def train(x_train, y_train, x_test, y_test, iters_train, loss_train, loss_val):

    max_iterations = 100
    num_samples, num_features = x_train.shape
    num_test_samples, num_test_features = x_test.shape
    theta = random.rand(num_features)
    gamma = 1

    lr = 0.01

    for i in range(max_iterations):
        iters.append(i)
        grad = 0
        for sample in range(num_samples):
            output = np.dot(x_train[sample], theta)
            diff = y_train[sample] - output
            each_grad = np.dot(x_train[sample], diff)
            grad += each_grad
        grad = (-2 * grad) / num_samples + gamma * theta

        theta -= lr * grad

        predict_error = 0
        for j in range(num_test_samples):
            predict_output = np.dot(x_test[j], theta)
            predict_error += np.dot((predict_output - y_test[j]), (predict_output - y_test[j])) + 0.5 * gamma * np.dot(theta, theta)
            predict_error /= num_test_samples
        print(str(i) + '\t' + str(predict_error))

        loss_train.append(predict_error)

        train_error = 0
        for j in range(num_samples):
            predict_output = np.dot(x_train[j], theta)
            train_error += np.dot((predict_output - y_train[j]), (predict_output - y_train[j])) + 0.5 * gamma * np.dot(theta, theta)
            train_error /= num_samples
        print(str(i) + '\t' + str(train_error))

        loss_val.append(train_error)

if __name__ == '__main__':
    x_train, y_train = ds.load_svmlight_file('./data/housing')

    x_train = x_train.toarray()
    temp = np.ones(shape=[506, 1], dtype=np.float32)
    x_train = np.concatenate(1[x_train, temp], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.367, random_state=42)

    num_samples, num_features = x_train.shape
    num_test_samples, num_test_features = x_test.shape

    iters = []
    test_errors = []
    train_errors = []
    train(x_train, y_train, x_test, y_test, iters, test_errors, train_errors)
    plt.plot(iters, test_errors, label='validation loss')
    plt.plot(iters, train_errors, label='training loss')

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
