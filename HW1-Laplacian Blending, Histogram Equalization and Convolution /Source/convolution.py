import numpy as np
import cv2


def fourierTransform(img, newSize=None):
    """
    A simple way to take fourier transform of a image.
    :param img: The image whose fourier transform should be returned.
    :param newSize: Reshape the image to a new shape.
    Possibly for masks to be reshaped to the same size as the image
    on which the mask is to be operated.
    :return: The fourier transform of the image.
    """
    dft = np.fft.fft2(img, newSize)
    return np.fft.fftshift(dft)


def invFourierTransform(maskedFreqDomain):
    """
    Perform inverse fourier transform.
    :param maskedFreqDomain: input image in fourier domain.
    :return: Image in spatial domain.
    """
    invFreqDomain = np.fft.ifftshift(maskedFreqDomain)
    img_back = np.fft.ifft2(invFreqDomain)
    return np.abs(img_back)


def deconvoluteImage(img):
    """
    HighPassFilter/LowPassFilter, send True if want to perform lowPassFilter operation on the image,
    otherwise pass False for high pass operation
    :param img:
    :return:
    """
    gk = cv2.getGaussianKernel(21, 5)
    gk = gk * gk.T
    imageFreqDomain = fourierTransform(img)
    gkFourierDomain = fourierTransform(gk, (img.shape[0], img.shape[1]))
    # apply mask and inverse DFT
    maskedFreqDomain = imageFreqDomain / gkFourierDomain
    return invFourierTransform(maskedFreqDomain)
