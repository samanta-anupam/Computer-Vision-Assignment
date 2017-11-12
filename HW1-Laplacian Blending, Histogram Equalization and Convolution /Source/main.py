# Instructions:
# For question 1, only modify function: histogram_equalization
# For question 2, only modify functions: low_pass_filter, high_pass_filter, deconvolution
# For question 3, only modify function: laplacian_pyramid_blending

import os
import sys
import cv2
import numpy
from histogram_equalization import equalizeImage, normalizeHist, calculateHist
from convolution import deconvoluteImage
from laplacian_blend import blendImage
from pass_filter import passFilter

def help_message():
    print("Usage: [Question_Number] [Input_Options] [Output_Options]")
    print("[Question Number]")
    print("1 Histogram equalization")
    print("2 Frequency domain filtering")
    print("3 Laplacian pyramid blending")
    print("[Input_Options]")
    print("Path to the input images")
    print("[Output_Options]")
    print("Output directory")
    print("Example usages:")
    print(sys.argv[0] + " 1 " + "[path to input image] " +
          "[output directory]")  # Single input, single output
    print(sys.argv[0] + " 2 " + "[path to input image1] " +
          "[path to input image2] " +
          "[output directory]")  # Two inputs, three outputs
    print(sys.argv[0] + " 3 " + "[path to input image1] " +
          "[path to input image2] " +
          "[output directory]")  # Two inputs, single output


# ===================================================
# ======== Question 1: Histogram equalization =======
# ===================================================

def histogram_equalization(img_in):
    """
    Method to perform histogram equalization of any image.
    :param img_in: The image that is to be equalized
    :return: equalized image and a stub success parameter.
    """
    channels = cv2.split(img_in)
    histEqImages = []
    for imageChannel in channels:
        eqImg = equalizeImage(imageChannel,
                              normalizeHist(calculateHist(imageChannel)))
        histEqImages.append(eqImg)

    img_out = cv2.merge(histEqImages)
    return True, img_out


def Question1():
    # Read in input images
    """
    Wrapper method to perform hisotgram equalization
    :return: success.
    """
    input_image = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)

    # Histogram equalization
    succeed, output_image = histogram_equalization(input_image)

    # Write out the result
    output_name = sys.argv[3] + "1.png"
    cv2.imwrite(output_name, output_image)

    return True


# ===================================================
# ===== Question 2: Frequency domain filtering ======
# ===================================================


def low_pass_filter(img_in):
    """
        A low pass filter. Simply remove the low frequency in the fourier transform of the image
        and Voila! we have a low pass filtered image. (Image with edges)
        :param img_in: Input image from which high frequency are remove.
        :return: low pass filtered image
        """
    channels = cv2.split(img_in)
    lowPassImage = []
    for imageChannel in channels:
        lowPassImage.append(passFilter(imageChannel, 20, True))
    img_out = cv2.merge(lowPassImage)
    return True, img_out


def high_pass_filter(img_in):
    """
    A high pass filter. Simply remove the high frequency in the fourier transform of the image
    and Voila! we have a high pass filtered image. (A blurred image)
    :param img_in: Input image from which high frequency are remove.
    :return: high pass filtered image.
    """
    channels = cv2.split(img_in)
    highPassImage = []
    for imageChannel in channels:
        # 20: bounding box of the frequency domain of the
        # image out of which higher frequencies are to be removed.
        # This value dictates how big is the mask, which removes the higher frequencies.
        highPassImage.append(passFilter(imageChannel, 20, False))
    img_out = cv2.merge(highPassImage)
    return True, img_out


def deconvolution(img_in):
    """
    Remove a known kernel from a image masked with a gaussian kernel
    Dividing the kernel in the frequency domain will remove the mask from
    the image
    :param img_in: The input image.
    :return: kernel free clear image.
    """
    channels = cv2.split(img_in)
    deconvolutedImage = []
    for imageChannel in channels:
        deconvolutedImage.append(deconvoluteImage(imageChannel * 255))
    img_out = cv2.merge(deconvolutedImage)
    return True, img_out


def Question2():

    # Read in input images
    input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)
    input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    # Low and high pass filter
    succeed1, output_image1 = low_pass_filter(input_image1)
    succeed2, output_image2 = high_pass_filter(input_image1)

    # Deconvolution
    succeed3, output_image3 = deconvolution(input_image2)

    # Write out the result
    output_name1 = sys.argv[4] + "2LPF.png"
    output_name2 = sys.argv[4] + "2HPF.png"
    output_name3 = sys.argv[4] + "2deconv.png"
    cv2.imwrite(output_name1, output_image1)
    cv2.imwrite(output_name2, output_image2)
    cv2.imwrite(output_name3, output_image3)

    return True


# ===================================================
# ===== Question 3: Laplacian pyramid blending ======
# ===================================================


def laplacian_pyramid_blending(img_in1, img_in2):

    # Write laplacian pyramid blending codes here
    """
    Laplacian blend two images.
    :param img_in1
    :param img_in2
    :return:
    """
    image1 = img_in1.copy()
    image2 = img_in2.copy()

    image1 = image1[:, :image1.shape[0]]
    image2 = image2[:image1.shape[0], :image1.shape[0]]

    img_out = blendImage(image1, image2)
    return True, img_out


def Question3():

    # Read in input images
    input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)
    input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_COLOR)

    # Laplacian pyramid blending
    succeed, output_image = laplacian_pyramid_blending(input_image1,
                                                       input_image2)

    # Write out the result
    output_name = sys.argv[4] + "3.png"
    cv2.imwrite(output_name, output_image)

    return True


if __name__ == '__main__':
    question_number = -1

    # Validate the input arguments
    if (len(sys.argv) < 4):
        help_message()
        sys.exit()
    else:
        question_number = int(sys.argv[1])

        if (question_number == 1 and not (len(sys.argv) == 4)):
            help_message()
            sys.exit()
        if (question_number == 2 and not (len(sys.argv) == 5)):
            help_message()
            sys.exit()
        if (question_number == 3 and not (len(sys.argv) == 5)):
            help_message()
            sys.exit()
        if (question_number > 3 or question_number < 1 or len(sys.argv) > 5):
            print("Input parameters out of bound ...")
            sys.exit()

    function_launch = {
        1: Question1,
        2: Question2,
        3: Question3,
    }

    # Call the function
    function_launch[question_number]()
