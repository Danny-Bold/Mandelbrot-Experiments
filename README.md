The Mandelbrot set is a famous fractal obtained by iterating a simple relation and seeing whether iterates remain bounded.

This project focusses on generating images of the Mandelbrot set and similar fractals:
-The Burning Ship Fractal is created by slightly altering the iteration schema and running the same process.
-The Buddhabrot is made by taking points that are not in the mandelbrot set and creating a 'probability density' of where these points land on their way to infinity.

Escape time rendering is used to colour fractals, with interpolation being added to allow user control over graduation of colours and colour selection.

Note that Buddhabrot generation is extremely slow due to 1000s of iterations per pixel, with CUDA atomic add operations necessary at every step.
