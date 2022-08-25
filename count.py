import numpy as np
import functools
a = np.array([[[1,2],[3,4]], [[5,6],[7,8]], [[9,10],[11,12]]])
index = 5
final = 0
for i in range(len(a.shape), 0, -1):
    print(i)
    if i - 1 >= 0:
        print(final, index)
        coord = (index  // (functools.reduce(lambda x, y: x * y, a.shape[i - 1::-1])))
        index = index - final

