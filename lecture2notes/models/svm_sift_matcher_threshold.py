# The purpose of this file is to find the best threshold value for the `min_match_count`
# value of `end_to_end/sift_matcher.match_features()`. The `yes` and `no` variables

import matplotlib.pyplot as plt
import numpy as np

# `pip install mlxtend`
from mlxtend.plotting import plot_decision_regions
from sklearn import svm

no = [52, 43, 42, 45, 62, 39, 40, 37, 22, 25, 32, 91, 32, 63, 11, 25, 11, 15, 21, 29]
yes = [
    63,
    65,
    61,
    56,
    100,
    54,
    48,
    37,
    18,
    30,
    50,
    84,
    180,
    193,
    137,
    80,
    89,
    93,
    104,
    99,
    17,
    31,
    41,
    32,
    45,
    41,
    43,
    64,
    63,
    42,
    49,
    44,
    65,
    28,
    20,
    23,
    33,
    48,
    26,
    38,
    45,
    46,
    58,
    50,
    43,
    74,
    43,
    47,
    37,
    35,
    46,
    48,
    46,
    49,
    40,
    24,
    63,
]

# Data balancing
no = np.repeat(no, 2)

y_no = np.zeros_like(no)
y_yes = np.ones_like(yes)

X = np.hstack((no, yes))
y = np.hstack((y_no, y_yes))

# plt.scatter(X,y)
# plt.show()

X_reshaped = X.reshape(-1, 1)

clf = svm.SVC(kernel="linear")
clf.fit(X_reshaped, y)

# `pip install mlxtend`
# Method from https://stackoverflow.com/a/58264443.
# Another method without the use of library is from https://stackoverflow.com/a/51301399.
plot_decision_regions(X_reshaped, y, clf=clf, legend=2)

plt.title("LinearSVC Decision Surface")
plt.ylabel("Class")
plt.xlabel("Number of Matches")

plt.show()
