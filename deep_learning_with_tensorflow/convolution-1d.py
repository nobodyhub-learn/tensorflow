import numpy as np

x1 = [3, 4, 5]
h1 = [2, 1, 0]

y1 = np.convolve(x1, h1)
print('y1:', y1)

x2 = [6, 2]
h2 = [1, 2, 5, 4]
print('y2', np.convolve(x2, h2, 'full'))

print('y3', np.convolve(x2, h2, 'valid'))
