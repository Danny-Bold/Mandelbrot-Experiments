import numpy as np
from numba import cuda
import matplotlib.pyplot as plt
import pygame

SIZE = (3840, 2160)

CENTRE = (-1.77, -0.035)

SCALE = 0.01

xRange = (CENTRE[0] - 8 * SCALE, CENTRE[0] + 8 * SCALE)
yRange = (CENTRE[1] - 4.5 * SCALE, CENTRE[1] + 4.5 * SCALE)

NUMBERSOFITERS = 1500


@cuda.jit
def burningShipIter(inputArray, out, maxIters):
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
            z = (abs(z.real) + 1j * abs(z.imag)) ** 2 + c

            if abs(z) < 10:
                itersToEscape += 1

            else:
                break

        if itersToEscape == maxIters:
            out[row, col] = -1

        else:
            out[row, col] = itersToEscape


def genInputs(array):
    for x in range(SIZE[0]):
        for y in range(SIZE[1]):
            array[x, y] = (xRange[1] - xRange[0]) * x / SIZE[0] + xRange[0] + \
                          1.0j * ((yRange[1] - yRange[0]) * y / SIZE[1] + yRange[0])


def genSingleColor(val):
    c = pygame.Color(0, 0, 0)
    c.hsva = (238, 100 * (1 - val), 100, 0)
    return c.r / 255, c.g / 255, c.b / 255


def lerp(x):
    return 1 - (1 - x) ** 50


def genColors(maxIters):
    colorList = []

    for x in range(maxIters):
        xAdjusted = lerp(x / maxIters)
        colorList.append(genSingleColor(xAdjusted))

    return colorList


def main():
    # Generate all different hues

    colorList = genColors(NUMBERSOFITERS)

    array = np.zeros(SIZE, dtype=np.csingle)

    genInputs(array)

    output = np.zeros(SIZE)

    threadsPerBlock = (32, 32)
    blocksPerGridX = (SIZE[0] + (threadsPerBlock[0] - 1)) // threadsPerBlock[0]
    blocksPerGridY = (SIZE[1] + (threadsPerBlock[1] - 1)) // threadsPerBlock[1]

    burningShipIter[(blocksPerGridX, blocksPerGridY), threadsPerBlock](array, output, NUMBERSOFITERS)

    img = np.zeros((*SIZE, 3))

    for x in range(SIZE[0]):
        for y in range(SIZE[1]):
            if output[x, y] == -1:
                img[x, y] = (0, 0, 0)

            else:
                img[x, y] = colorList[int(output[x, y]) % len(colorList)]

    plt.imsave('burningShip/' + str(NUMBERSOFITERS) + 'itersWithColZoomed.png', img.swapaxes(0, 1))


if __name__ == '__main__':
    main()
