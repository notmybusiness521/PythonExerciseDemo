# --*--coding=utf8--*--
'''
@Author:
@Time:
@Describe:
'''
import numpy as np
from scipy.spatial.distance import pdist

def main():
    np.random.seed(10)
    x = np.random.random(5)
    y = np.random.random(5)
    d1 = np.dot(x, y)/ (np.linalg.norm(x) * np.linalg.norm(y))
    X = np.vstack([x, y])
    d2 = 1 - pdist(X, "cosine")
    print(d1, d2)


if __name__=="__main__":
    main()