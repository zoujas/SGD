import numpy as np
import matplotlib.pyplot as plt


# Given training and testing data, learning rate epsilon, and a specified batch size,
# conduct stochastic gradient descent (SGD) to optimize the weight matrix W (785x10).
# Then return W.
def softmaxRegression(trainingImages, trainingLabels, testingImages, testingLabels, epsilon=None, batchSize=None):
    row,col = testingImages.shape
    row2,col2 = trainingLabels.shape
    w = np.random.randn(row, col2)

    shuffled_trainingImages, shuffled_trainingLables = unison_shuffle(trainingImages.transpose(), trainingLabels)
    shuffled_trainingImages = shuffled_trainingImages.transpose()

    epoch = 600

    minibatches = len(trainingLabels) / batchSize
    for ep in range(epoch):
        for i in range(minibatches - 1):
            minibatch = shuffled_trainingImages[:, batchSize * i: batchSize * (i + 1)]
            label_batch = shuffled_trainingLables[ batchSize * i: batchSize * (i + 1)]


            z = minibatch.transpose().dot(w)
            expo_z = np.exp(z)
            y_hat = expo_z / np.sum(expo_z,axis= 1).reshape(-1,1)
            print(ep,i,CE_loss(label_batch,y_hat))
            gradient = minibatch.dot((y_hat - label_batch))
            gradient = gradient/len(minibatch)
            w = w - epsilon * gradient

    final_z = testingImages.transpose().dot(w)

    final_expo_z = np.exp(final_z)

    final_y_hat = final_expo_z/np.sum(final_expo_z,axis = 1).reshape(-1,1)

    orig_testingLables = np.argmax(testingLabels,1)
    orig_y_hat = np.argmax(final_y_hat,1)


    acc = PC_accuracy(orig_testingLables,orig_y_hat)
    ce_loss = CE_loss(testingLabels,final_y_hat)

    return w


def PC_accuracy(y,y_hat):
    correct_guesses = np.sum(y_hat == y)
    return float(correct_guesses)/len(y_hat)

def CE_loss(y,yhat):
    yhat = np.log(yhat)
    sum = np.sum(np.multiply(y,yhat))
    sum = sum/(-1*len(y))

    return sum





def one_hot(lables):
    lables2 = np.zeros((lables.size, lables.max() + 1))
    lables2[np.arange(lables.size), lables] = 1
    return lables2


def unison_shuffle(images, lables,):
    assert len(images) == len(lables)
    p = np.random.permutation(len(images))
    return images[p], lables[p]


if __name__ == "__main__":
    # Load data
    trainingImages = np.load("fashion_mnist_train_images.npy")
    trainingLabels = np.load("fashion_mnist_train_labels.npy")
    testingImages = np.load("fashion_mnist_test_images.npy")
    testingLabels = np.load("fashion_mnist_test_labels.npy")

    # Append a constant 1 term to each example to correspond to the bias terms
    # ...

    trainingImages = np.divide(trainingImages, 255.)
    testingImages = np.divide(testingImages,255.)
    trainingImages = trainingImages.transpose()
    trainingImages = np.vstack((trainingImages, np.ones(60000)))
    testingImages = testingImages.transpose()
    row,col = testingImages.shape

    testingImages  = np.vstack((testingImages,np.ones(col)))

    trainingLabels = one_hot(trainingLabels)
    testingLabels = one_hot(testingLabels)

    row, col = trainingImages.shape

    W = softmaxRegression(trainingImages, trainingLabels, testingImages, testingLabels, epsilon=0.1, batchSize=100)



    W = W[:-1]
    W= W.transpose()
    """ print(W[0])
    plt.imshow(W[0].reshape(28, 28))
    plt.show()
    plt.imshow(W[1].reshape(28,28))
    plt.show()
    plt.imshow(W[2].reshape(28, 28))
    plt.show()
    plt.imshow(W[3].reshape(28,28))
    plt.show()
    plt.imshow(W[4].reshape(28, 28))
    plt.show()
    plt.imshow(W[5].reshape(28, 28))
    plt.show()
    plt.imshow(W[6].reshape(28, 28))
    plt.show()
    plt.imshow(W[7].reshape(28, 28))
    plt.show()
    plt.imshow(W[8].reshape(28, 28))
    plt.show()
    plt.imshow(W[9].reshape(28, 28))
    plt.show()
    """




    # Visualize the vectors
    # ...
