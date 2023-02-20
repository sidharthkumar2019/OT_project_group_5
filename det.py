
import numpy as np

arr = np.array([
        [0,0,1,1,3],
        [0,0,5,2,1],
        [1,5,2,0,0],
        [2,2,0,2,0],
        [3,1,0,0,2]
    ])
det = np.linalg.det(arr)
print(det)
