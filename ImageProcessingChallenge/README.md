Setting up the Python code environment
=========================================
This code assumes that python3 is already installed on your Linux Machine.

Creating Virtual Environment
=============================
1. cd ImageProcessingChallenge
1. sudo apt install python3-venv
2. python3 -m venv my-project-env
3. source my-project-env/bin/activate
4. (my-project-env) $
This above line means you are in your virtual environment.

Install the dependencies in the virtual environment.
5. pip install opencv-python


Running the Python code
=============================
6. python3 src.py

To view the results
=============================
7. open writeup.html

Image Processing Techniques covered in the code
==============================
1. Change Contrast of the image
2. Brighten the image
3. Blur the given image
4. Sharpen the image
5. Detect the edges in the image
6. Change saturation level of the image
7. Composite operation: This operation composites the top image over the base image, using the alpha channel of the top image as a mask.

References
======================================
1. http://www.graficaobscura.com/interp/index.html
2. https://www.cs.princeton.edu/courses/archive/spring14/cos426/assignment1/examples.html
3. https://learnopencv.com/alpha-blending-using-opencv-cpp-python/
4. https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
