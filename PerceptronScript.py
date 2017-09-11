import time


############################################################################################################

import pandas as pd

while True:
    var = input("\nBegin Perceptron Script? (Y/N) ")
    if var == 'N':
        quit()
    elif var == 'Y':
        break



df = pd.read_csv('https://archive.ics.uci.edu/ml/' 'machine-learning-databases/iris/iris.data', header=None)
print('\nSuccessfully imported Iris dataset\n')
time.sleep(2)

print('\nLast five samples of Iris dataset (for confirmation):')
print(df.tail())
time.sleep(5)


############################################################################################################


import matplotlib.pyplot as plt
import numpy as np

# select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
print('\nAssigned Setosa = -1 and Versicolor = 1\n')

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values
print('\nExtracted sepal length and petal length\n')

# plot data
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

print('\nPlotting: Iris subset (first 100 samples) dataplot\n')
print('(close Figure to continue)')

plt.tight_layout()
#plt.savefig('./images/02_06.png', dpi=300)
plt.show()


time.sleep(2)


############################################################################################################


print('\nPreparing to train the Perceptron\n')

time.sleep(2)

from Perceptron import Perceptron

print('\nSuccessfully imported my Perceptron class\n')

ppn = Perceptron(eta=0.1, n_iter=10)

ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')

print('\nPlotting: Misclassification vs Epochs plot (for convergence)\n')
print('(close Figure to continue)')

plt.tight_layout()
# plt.savefig('./perceptron_1.png', dpi=300)
plt.show()

time.sleep(5)


############################################################################################################


from matplotlib.colors import ListedColormap

print('\nPreparing to create decision-regions plot\n')

time.sleep(2)

def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                         np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples again
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)


plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

print('\nPlotting: Decision-regions plot\n')
print('(close Figure to end program)')

plt.tight_layout()
# plt.savefig('./perceptron_2.png', dpi=300)
plt.show()



def newTest():
    sepalLength = input("\nEnter sepal length: ")
    sepalWideth = input("\nEnter sepal width: ")

    if (sepalLength > X) and sepalWidth > Y):
        print('Result: Setosa')
    else:
        print('Result: Versicolor')

    # TO-DO: Print whether this new input dimensions correspond to Setosa or Versicolor


while True:
    var = input("\nTest with new data point? (Y/N) ")
    if var == 'N':
        quit()
    elif var == 'Y':
        newTest()




print('\nProgram ended')
