from scipy import signal as sg

I = [[255, 7, 3],
     [212, 240, 4],
     [218, 216, 230]]

g1 = [[-1, 1]]

print('without zero padding: \n{0}\n'.format(sg.convolve(I, g1, 'valid')))
print('with zero padding: \n{0}\n'.format(sg.convolve(I, g1)))

g2 = [[-1, 1],
      [2, 3]]

print('without zero padding: \n{0}\n'.format(sg.convolve(I, g2, 'valid')))
print('with zero padding: \n{0}\n'.format(sg.convolve(I, g2)))
