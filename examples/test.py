from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn_oblique_tree.oblique import ObliqueTree
import numpy as np
import polytope as pc
import matplotlib.pyplot as plt

random_state = 2

#see Murthy, et all for details.
#For oblique with consideration of axis parallel
# tree = ObliqueTree(splitter="oc1, axis_parallel", number_of_restarts=20, max_perturbations=5, random_state=random_state)
#
#For multivariate CART select 'cart' splitter
# tree = ObliqueTree(splitter="cart", number_of_restarts=20, max_perturbations=5, random_state=random_state)

#consider only oblique splits
tree = ObliqueTree(splitter="oc1", number_of_restarts=20, max_perturbations=5, random_state=random_state)

# X_train = np.random.random_sample((100, 2))
# y_train = np.random.choice([0, 1], 100)

X_train, X_test, y_train, y_test = train_test_split(*load_iris(return_X_y=True), test_size=.4, random_state=random_state)
print(X_train.shape)
print(y_train)
tree.fit(X_train, y_train)

predictions = tree.predict(X_test)

partition = tree.get_partition()
print('num regions:')
print(len(partition))

# # plot partition
# polys = []
# for r in partition:
#     polys.append(
#         pc.Polytope(r[:, :-1], r[:, -1])
#     )
# for p in polys:
#     p.plot()
# plt.show()
print("Iris Accuracy:",accuracy_score(y_test, predictions))