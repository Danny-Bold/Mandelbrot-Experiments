import numpy as np
from numba import cuda
import matplotlib.pyplot as plt
import pygame
import imageio.v3 as iio

SIZE = (3840, 2160)

xRange = (-2.8125, 1.875)
yRange = (-1.25, 1.25)

NUMBERSOFITERS = 50

pointsPerPixel = 100  # TODO crank up PPP and see if more patterns emerge - use 1-(1-x)^n interpolation to highlight areas more




@cuda.jit
def divideByMax(array, out, maxVal):
    row, col = cuda.grid(2)

    if row < array.shape[0] and col < array.shape[1]:
        val = array[row, col] / maxVal

        interp = 1 - (1 - val) ** 50

        for x in range(3):
            out[row, col, x] = interp


@cuda.jit
def buddhabrotIter(output, mandel, pointsPerPixel, xR, yR, maxIters):
    row, col = cuda.grid(2)

    if row < output.shape[0] and col < output.shape[1]:
        """
        
        Produce grid of points - pointsPerPixel refers to the side length of the grid rather than the number of points.
        
        The region we're checking for to increment counter by one is:
        
        row * (xRange[1] - xRange[0]) / SIZE[0] + xRange[0] <= z.real < (row + 1) * (xRange[1] - xRange[0]) / SIZE[0] + xRange[0])
        col * (yRange[1] - yRange[0]) / SIZE[1] + yRange[0] <= z.imag < (col + 1) * (yRange[1] - yRange[0]) / SIZE[1] + yRange[0])
        
        """
        if mandel[row, col, 0] != 0:  # Then point isn't in mandelbrot set, generate points

            xVal, yVal = row * (xR[1] - xR[0]) / SIZE[0] + xR[0], \
                         col * (yR[1] - yR[0]) / SIZE[1] + yR[0]

            for x in range(pointsPerPixel):
                for y in range(pointsPerPixel):
                    xProp, yProp = x / pointsPerPixel, y / pointsPerPixel  # proportion of distance across pixel
                    pixelOffsetX, pixelOffsetY = xProp * (xR[1] - xR[0]) / SIZE[0], \
                                                 yProp * (yR[1] - yR[0]) / SIZE[1]

                    # Calculate position of pixel within grid

                    xPos = xVal + pixelOffsetX
                    yPos = yVal + pixelOffsetY

                    c = xPos + 1j * yPos

                    z = 0

                    for iteration in range(maxIters):

                        z = z ** 2 + c

                        x = z.real

                        y = z.imag
                        # Could have said 'if x in xRange and y in yRange'
                        pixelUnrounded = int((x - xR[0]) * SIZE[0] / (xR[1] - xR[0])), int((y - yR[0]) * SIZE[1] / (yR[1] - yR[0]))  # magic

                        pixel = int(round(pixelUnrounded[0])), int(round(pixelUnrounded[1]))

                        if 0 <= pixel[0] < SIZE[0] and 0 <= pixel[1] < SIZE[1]:
                            cuda.atomic.add(output, pixel, 1)


def main():
    array = np.zeros(SIZE)

    mandelbrot = np.ascontiguousarray(iio.imread('mandelbrot/500Iters.png').swapaxes(0, 1))

    threadsPerBlock = (32, 32)
    blocksPerGridX = (SIZE[0] + (threadsPerBlock[0] - 1)) // threadsPerBlock[0]
    blocksPerGridY = (SIZE[1] + (threadsPerBlock[1] - 1)) // threadsPerBlock[1]
    blocksPerGrid = (blocksPerGridX, blocksPerGridY)

    buddhabrotIter[blocksPerGrid, threadsPerBlock](array, mandelbrot, pointsPerPixel, xRange, yRange, NUMBERSOFITERS)

    scaled = np.zeros((*SIZE, 3))

    maxVal = np.max(array)

    divideByMax[blocksPerGrid, threadsPerBlock](array, scaled, maxVal)

    print(scaled)

    plt.imsave('buddhabrotOld/greyScale' + str(NUMBERSOFITERS) + 'PPP' + str(pointsPerPixel) + '.png', scaled)


if __name__ == '__main__':
    main()


"""

PPP = 100 ran for 6hr25min and didn't produce an output.

TODO: Rewrite in two steps so that atomicAdd doesn't halt performance.
1) buddhaBrotIter fills in matrix with shape (*SIZE, NUMBEROFITERS * pointsPerPixel, 2) with each entry being the index of the cell to be filled in
2) Second function gathers the data and collates it into one array that can be turned into an image.

This wont work, since for 500 iterations and 100 PPP this would require 603 TB :(

"""