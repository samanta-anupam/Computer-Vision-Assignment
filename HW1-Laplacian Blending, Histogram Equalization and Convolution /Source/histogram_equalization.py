import cv2
import numpy as np


def calculateHist(imageChannel):
    """
    Calculate histogram of the image for each channel.
    :param imageChannel: Pass each of the channel of the image i.e R,G,B and then
    calculate the histogram of each image.
    :return: histogram array.
    """
    x = range(256)
    # Calculate the number of pixels for each pixel value from 0 to 255
    # Passing range here is important otherwise, np.histogram creates
    # a range from the first non zero value to the last non zero value.
    y, bins = np.histogram(imageChannel, bins=256, range=(0, 255))
    return y


def normalizeHist(hist):
    """
    Normalize the frequency of each pixel by the total number of pixels.
    :param hist: The histogram of one image.
    :return: cumulative distribution function of the image
    """
    cdf = np.cumsum(hist)
    n = cdf[-1]
    cdf = cdf.astype(float)
    cdf = (cdf * 255 / n)
    cdf = cdf.astype(int)
    return cdf


def equalizeImage(imageChannel, cdf):
    """
    Apply the normalized histogram so obtained from normalizeHist to each image channels
    and obtain the final equalized image.
    :param imageChannel: each of the channels of the image.
    :param cdf: cdf of the image as obtained by the function of the normalizeHist function.
    :return: The final result, with equalized histogram of the image.
    """
    for i in range(len(imageChannel)):
        for j in range(len(imageChannel[0])):
            imageChannel[i][j] = cdf[imageChannel[i][j]]
    return imageChannel
