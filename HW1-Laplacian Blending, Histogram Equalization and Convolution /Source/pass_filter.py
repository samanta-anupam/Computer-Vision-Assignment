import numpy as np
import cv2


def fourierTransform(img):
    """
    Calculate the fourier transform of the image.
    :param img: input image
    :return: fourier transform of the image
    """
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    return dft_shift


def invFourierTransform(maskedFreqDomain):
    """
    Inverse fourier transform of the fourier domain of the image.
    :param maskedFreqDomain: fourier domain of the image
    :return: Return the image.
    """
    shiftedFourierDomain = np.fft.ifftshift(maskedFreqDomain)
    spatialDomain = cv2.idft(shiftedFourierDomain, flags=cv2.DFT_SCALE)
    img_back = cv2.magnitude(spatialDomain[:, :, 0], spatialDomain[:, :, 1])
    return img_back


def passFilter(img, mask_size, lowPass=False):
    """
    HighPassFilter/LowPassFilter, send True if want to perform lowPassFilter operation on the image,
    otherwise pass False for high pass operation
    :param img: input image
    :param mask_size: mask to be applied
    :param lowPass: Whether it should be used as a high pass or low pass filter.
    :return: High/Low pass filtered image
    """
    freqDomain = fourierTransform(img)
    x, y = img.shape
    if lowPass:
        mask = np.zeros((x, y, 2), np.uint8)
        mask[x / 2 - mask_size / 2:x / 2 + mask_size / 2, y / 2 - mask_size / 2:
        y / 2 + mask_size / 2] = 1
    else:
        mask = np.ones((x, y, 2), np.uint8)
        mask[x / 2 - mask_size / 2:x / 2 + mask_size / 2, y / 2 - mask_size / 2:
        y / 2 + mask_size / 2] = 0

    # apply mask and inverse DFT
    maskedFreqDomain = freqDomain * mask
    return invFourierTransform(maskedFreqDomain)
