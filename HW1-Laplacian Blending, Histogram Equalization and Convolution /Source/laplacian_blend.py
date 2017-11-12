import numpy as np
import cv2

# Using only 4 height gaussian pyramid.
DOWNSAMPLE_NUM = 4


def gaussianPyramid(img):
    """
    Compute the gaussian pyramid of the image.
    A gaussian pyramid is a pyramid of images with the base being the highest resolution,
    and higher up layers are gaussian blurs of the successive previous layer.
    :param img: The image whose gaussian pyramid is to be built
    :return: array of images, that represent the gaussian pyramid.
    """
    gausPyr = [img]
    for i in range(DOWNSAMPLE_NUM):
        img = cv2.pyrDown(img)
        gausPyr.append(img)
    return gausPyr


def laplacianPyramid(img, gaus):
    """
    Laplacian pyramid contains the higher frequency between two layers that has been lost in
    the upper layer of the gaussian pyramid. Useful when stitching, where we simply add these
    details to the apex of the gaussian pyramid to create a perfect stitched photo.
    :param img: input image
    :param gaus: gaussian pyramid of the image.
    :return: laplacian pyramid of the image.
    """
    lapPyr = [gaus[-1]]
    for i in range(len(gaus) - 1, 0, -1):
        prevGaus = cv2.pyrUp(gaus[i])
        temp = cv2.subtract(gaus[i - 1], prevGaus)
        lapPyr.append(temp)
    return lapPyr


def stitchImages(img1, img2):
    """
    Images that are to be stitched half and half.
    You could do funky things here by adding masks and stuff.
    :param img1: input image1
    :param img2: input image2,
    :return: the blended image.
    """
    stitchedImage = []
    for l1, l2 in zip(img1, img2):
        rows, cols, h = l1.shape
        temp = np.hstack((l1[:, 0:cols / 2], l2[:, cols / 2:]))
        stitchedImage.append(temp)
    return stitchedImage


def blendImage(image1, image2):
    """
    Blend two images. Takes two params, image1 and image2 and then blends by using gaussian
    and laplacian blending. Laplacian pyramid are useful for blending two images, by
    progressively adding high frequency to each layer as we start from the apex of
    the gaussian pyramid and start adding higher frequencies from the equivalent gaussian pyramid.
    :param image1:input1
    :param image2:input2
    :return:
    """
    gausPyr1 = gaussianPyramid(image1)
    gausPyr2 = gaussianPyramid(image2)

    lapPyr1 = laplacianPyramid(image1, gausPyr1)
    lapPyr2 = laplacianPyramid(image2, gausPyr2)

    stitchedLaplacian = stitchImages(lapPyr1, lapPyr2)

    finalImage = stitchedLaplacian[0]
    for i in range(1, len(stitchedLaplacian)):
        finalImage = cv2.add(stitchedLaplacian[i], cv2.pyrUp(finalImage))
    return finalImage
