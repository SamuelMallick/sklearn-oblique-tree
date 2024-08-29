from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn_oblique_tree.oblique import ObliqueTree
import numpy as np
import polytope as pc
import matplotlib.pyplot as plt

random_state = 2
np_random = np.random.default_rng(4)

#see Murthy, et all for details.
#For oblique with consideration of axis parallel
# tree = ObliqueTree(splitter="oc1, axis_parallel", number_of_restarts=20, max_perturbations=5, random_state=random_state)
#
#For multivariate CART select 'cart' splitter
# tree = ObliqueTree(splitter="cart", number_of_restarts=20, max_perturbations=5, random_state=random_state)

#consider only oblique splits
tree = ObliqueTree(splitter="oc1", number_of_restarts=20, max_perturbations=5, random_state=random_state)

num_points = 500
X_train = 2*np_random.random((num_points, 2))-1
y_train = np.array([0 if x[1]-x[0] <= 0 else 1 for x in X_train]).reshape(-1, 1)
A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
b = np.array([[1, 1, 1, 1]]).T
fig, ax = plt.subplots()
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
ax.plot(X_train[y_train.squeeze() == 0][:, 0], X_train[y_train.squeeze() == 0][:, 1], "r.", markersize=1)
ax.plot(X_train[y_train.squeeze() == 1][:, 0], X_train[y_train.squeeze() == 1][:, 1], "g.", markersize=1)

# X_train, X_test, y_train, y_test = train_test_split(*load_iris(return_X_y=True), test_size=.4, random_state=random_state)
# print(X_train.shape)
# print(y_train)
tree.fit(X_train, y_train)

# predictions = tree.predict(X_test)

partition = tree.get_partition()
print('num regions:')
print(len(partition))

# plot partition
fig, ax = plt.subplots()
polys = []
for r in partition:
    polys.append(
        pc.Polytope(np.vstack((r[:, :-1], A)), np.vstack((r[:, [-1]], b)))
    )
for p in polys:
    p.plot(ax = ax)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.show()
# print("Iris Accuracy:",accuracy_score(y_test, predictions))