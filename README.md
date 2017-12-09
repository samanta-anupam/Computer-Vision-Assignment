# Computer-Vision-Assignment
Assignment as given in the course work of Introduction to Computer Vision(CSE527) taken by Prof Roy Shilkrot

## The goal in this assignment was filtering in the spatial domain as well as in the frequency domain.

Laplacian Blending using Image Pyramids is a very good intro to working and thinking in frequencies, and Deconvolution is a neat trick.

Tasks:
Perform Histogram Equalization on the given input image.
Perform Low-Pass, High-Pass and Deconvolution on the given input image.
Perform Laplacian Blending on the two input images (blend them together).

Histogram Equalization
Refer to the readings on @43, particularly to Szeliski's section 3.4.1, and within it to eqn 3.9.
Getting the histogram of a grayscale image is incredibly easy (Python):

Low-Pass, High-Pass and Deconvolution in the Frequency Domain
http://docs.opencv.org/master/de/dbc/tutorial_py_fourier_transform.html
http://docs.opencv.org/2.4/doc/tutorials/core/discrete_fourier_transform/discrete_fourier_transform.html

LPF: mask a 20x20 window of the center of the FT image (the low frequencies).
HPF: just reverse the mask.

Deconvolution: apply a gaussian kernel (gk) to your input image (in the FD/FFT):

Laplacian Pyramid Blending
http://docs.opencv.org/3.2.0/dc/dff/tutorial_py_pyramids.html


## HW2: Image Alignment, Panoramas

The goal was to create 2 panoramas:
Using homographies and perspective warping on a common plane (3 images).
Using cylindrical warping (many images).
In both options we should:
Read in the images: input1.jpg, input2.jpg, input3.jpg
[Apply cylindrical wrapping if needed]
Calculate the transformation (homography for projective; affine for cylindrical) between each
Transform input2 and input3 to the plane of input1, and produce output.png
Use Laplacian Blending code to stitch the images together nicely



## HW3: Detection and Tracking
Goal is to:
Detect the face in the first frame of the movie
Using pre-trained Viola-Jones detector
Track the face throughout the movie using:
CAMShift
Particle Filter
Face detector + Kalman Filter (always run the kf.predict(), and run kf.correct() when we get a new face detection)

Face Detector + Optical Flow tracker (use the OF tracker whenever the face detector fails).


## HW4: Segmentation
Perform semi-automatic binary segmentation based on SLIC superpixels and graph-cuts:

Given an image and sparse markings for foreground and background
Calculate SLIC over image
Calculate color histograms for all superpixels
Calculate color histograms for FG and BG
Construct a graph that takes into account superpixel-to-superpixel interaction (smoothness term), as well as superpixel-FG/BG interaction (match term)
Run a graph-cut algorithm to get the final segmentation

"Wow factor":
Make it interactive: Let the user draw the markings
for every interaction step
recalculate only the FG-BG histograms,
construct the graph and get a segmentation from the max-flow graph-cut,
show the result immediately to the user (should be fast enough).


## HW5: Structured Light

Goal is to reconstruct a scene from multiple structured light scannings of it.

Calibrate projector with the “easy” method
Use ray-plane intersection
Get 2D-3D correspondence and use stereo calibration
Get the binary code for each pixel - this you should do, but it's super easy
Correlate code with (x,y) position - we provide a "codebook" from binary code -> (x,y)
With 2D-2D correspondence
Perform stereo triangulation (existing function) to get a depth map

Add color to 3D cloud
When finding correspondences, take the RGB values from "aligned001.png"
Add them later to reconstruction
Output a file called "output_color.xyzrgb" with the following format
"%d %d %d %d %d %d\n"%(x, y, z, r, g, b)
for each 3D+RGB point
