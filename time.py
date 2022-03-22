import os
import time
import numpy as np

result = []
for i in np.linspace(1, 30000, 10):
    num_atoms = i
    start = time.time()
    os.system('./bond {}'.format(num_atoms))
    end = time.time()
    result.append([num_atoms*200, end-start])

print(result)
