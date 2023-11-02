import glob

import numpy as np
from numba import cuda
import matplotlib.pyplot as plt
import cv2

SIZE = (3840, 2160)

xRange = (-2.8125, 1.875)
yRange = (-1.25, 1.25)

NUMBERSOFITERS = 500


def videoGen():
    img_array = []
    filenames = glob.glob('mandelbrotVidImg/*.png')

    for x in range(len(filenames)):
        img = cv2.imread('mandelbrotVidImg/' + str(x) + '.png')
        img_array.append(img)

    out = cv2.VideoWriter('mandelbrotVid/vid.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 60, SIZE)

    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()


@cuda.jit
def mandelBrotIter(inputArray, out, maxIters):
    row, col = cuda.grid(2)

    if row < inputArray.shape[0] and col < inputArray.shape[1]:
        """
        Iterate maxIters times:
        do z^2+c op
        if not escaped:
            itersToEscape += 1

        else:
            break

        after loop, set out[row, col] = itersToEscape

        Then we're left with a matrix where out[i, j] = maxIters -> black pixel
        Otherwise, colour pixel accordingly by value.

        """

        itersToEscape = 0

        c = inputArray[row, col]

        z = 0

        for x in range(maxIters):
            z = z ** 2 + c

            if abs(z) < 2:
                itersToEscape += 1

            else:
                break

        if itersToEscape == maxIters:
            out[row, col] = (0, 0, 0)

        else:
            out[row, col] = (1, 1, 1)


def genInputs(array):
    for x in range(SIZE[0]):
        for y in range(SIZE[1]):
            array[x, y] = (xRange[1] - xRange[0]) * x / SIZE[0] + xRange[0] + \
                          1.0j * ((yRange[1] - yRange[0]) * y / SIZE[1] + yRange[0])


def main():

    for iteration in range(NUMBERSOFITERS):

        array = np.zeros(SIZE, dtype=np.csingle)

        genInputs(array)

        output = np.zeros((*SIZE, 3))

        threadsPerBlock = (32, 32)
        blocksPerGridX = (SIZE[0] + (threadsPerBlock[0] - 1)) // threadsPerBlock[0]
        blocksPerGridY = (SIZE[1] + (threadsPerBlock[1] - 1)) // threadsPerBlock[1]

        mandelBrotIter[(blocksPerGridX, blocksPerGridY), threadsPerBlock](array, output, iteration)

        plt.imsave('mandelbrotVidImg/' + str(iteration) + '.png', output.swapaxes(0, 1))

        print(iteration, 'done')

    videoGen()


if __name__ == '__main__':
    main()
