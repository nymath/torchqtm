# main.py
import numpy as np
from mysearch import sort_vector

arr = np.array([5, 3, 1, 4, 2], dtype=np.int32)
print("Before sorting:", arr)
sort_vector(arr)
print("After sorting:", arr)
