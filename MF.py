from math import *
import numpy as np
import matplotlib.pyplot as plt


def matrix_factorization(Y, rank, aplha, beta, steps):
    '''
    :param Y: label matrix m*n
    :param U: Linear features of miRNAs m*k
    :param V: Linear features of diseases m*k
    :param K: The dimension of the linear feature
    :param aplha: Learning rate
    :param beta: Regularization parameters
    :param steps:
    :return:
    '''
    print('Begin to decompose the original matrix: \n')

    Y = np.array(Y)

    # Number of rows of label matrix Y
    rows_Y = len(Y)

    # Number of columns of label matrix Y
    columns_Y = len(Y[0])  # The number of columns of the original matrix R

    # Random initialization matrix. [0 1]
    U = np.random.rand(rows_Y, rank)
    print(U)
    print(U.shape)
    V = np.random.rand(columns_Y, rank)
    # Transpose
    V = V.T

    result = []

    # update parameters using gradient descent method
    print('Start training: \n')
    for step in range(steps):
        for i in range(len(Y)):
            for j in range(len(Y[0])):
                eij = Y[i][j] - np.dot(U[i, :], V[:, j])
                for k in range(rank):
                    if Y[i][j] > 0:
                        # update parameters
                        U[i][k] = U[i][k] + aplha * (2 * eij * V[k][j] - beta * U[i][k])
                        V[k][j] = V[k][j] + aplha * (2 * eij * U[i][k] - beta * V[k][j])

        # loss
        e = 0
        for i in range(len(Y)):
            for j in range(len(Y[i])):
                if Y[i][j] > 0:
                    e = e + pow(Y[i][j] - np.dot(U[i, :], V[:, j]), 2)  # loss
                    for k in range(rank):
                        e = e + (beta / 2) * (pow(U[i][k], 2) + pow(V[k][j], 2))  # loss with regularization
        result.append(e)
        if e < 0.001:
            break
    print('training Finshed 。。。。')

    return U, V.T, result


if __name__ == '__main__':  # Main
    Y = np.load('.\data\HMDD v3.2\miRNA-disease association.npy')
    Y = np.array(Y)

    U, V, result = matrix_factorization(Y, 50, aplha=0.021, beta=0.00002, steps=100)
    np.save('./data/U.npy',arr=U) # Linear characterization of miRNAs
    np.save('./data/V.npy',arr=V) # Linear characterization of diseases


    # show the result
    print(result)
    print(result[99])
    print('Original matrix: ', Y)
    print(U)
    print(V)
    Y_MF = np.dot(U, V.T)
    print('The calculated matrix: ', Y_MF)

    # display the results with a chart
    plt.plot(range(len(result)), result)
    plt.xlabel("time")
    plt.ylabel("loss")
    plt.show()