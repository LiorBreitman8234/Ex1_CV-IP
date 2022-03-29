"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List
import cv2
import numpy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 212733257


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    if representation == 1:
        im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        mat = np.asarray(im, np.float)
        mat = mat / 255
    else:
        im = cv2.imread(filename, cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        mat = np.asarray(im, np.float)
        mat = mat / 255
    return mat


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    im = imReadAndConvert(filename, representation)
    if representation == 1:
        plt.imshow(im, cmap='gray')
    else:
        plt.imshow(im)
    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    r, g, b = cv2.split(imgRGB)
    # transformationMatrix = np.array([[0.299, 0.576, 0.144], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
    # test = np.dot(transformationMatrix, imRGB)
    y = r * 0.299 + 0.576 * g + 0.144 * b
    i = r * 0.596 - 0.275 * g - 0.321 * b
    q = 0.212 * r - 0.523 * g + 0.311 * b
    mat = np.array([y, i, q])
    mat = np.moveaxis(mat, 0, -1)
    return mat


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    y, i, q = cv2.split(imgYIQ)
    # transformationMatrix = np.array([[1,0.956, 0.619],[1,-0.272,-0.647],[1,-1.106,1.703]])
    # test = np.dot(transformationMatrix, imRGB)
    r = y * 1 + 0.956 * i + 0.619 * q
    g = y * 1 - 0.272 * i - 0.647 * q
    b = y * 1 - 1.106 * i + 1.703 * q
    mat = np.array([r, g, b])
    mat = np.moveaxis(mat, 0, -1)
    return mat


def hsitogramEqualize(imOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imOrig: Original Histogram
        :ret
    """

    norm_im = cv2.normalize(imOrig, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_im = norm_im.astype(int)
    origHist = np.histogram(norm_im, bins=256)
    cumSum = np.cumsum(origHist[0])
    norm_cumSum = (cumSum / cumSum.max())
    LUT = np.ceil(norm_cumSum * 255)
    new_im = numpy.copy(imOrig)
    for i in range(256):
        new_im[norm_im == i] = LUT[i]
    histNew = np.histogram(new_im, bins=256)
    new_im = new_im / 255
    return new_im, origHist[0], histNew[0]


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    # initial bounds
    workOn = imOrig
    flag = 0
    if len(imOrig.shape) == 3:
        imYIQ = transformRGB2YIQ(imOrig)
        y,i,q = cv2.split(imYIQ)
        workOn = y
        flag = 1

    boundSize = float(255) / float(nQuant)
    print(boundSize)
    bounds = np.arange(0, 256 - boundSize, np.floor(boundSize), dtype = int)
    bounds = np.append(bounds, 255)
    # normalizing picture and using histogram
    norm_im = cv2.normalize(workOn, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_im = norm_im.astype(int)
    hist = np.histogram(workOn, bins=256)
    means = np.zeros(nQuant)
    pics = []
    errors = []
    # the algo itself
    for j in range(nIter):
        for i in range(nQuant):
            means[i] = hist[2][np.floor(np.mean(hist[0][bounds[i]:bounds[i+1]]))]
        runOver = np.arange(1, len(means))
        for i in runOver:
            print(i)
            print(bounds[i])
            print(means[i])
            bounds[i] = np.floor((means[i-1] + means[i])/2)
        mse = np.sqrt(np.square(np.sum(imOrig - means.mean()))) / (imOrig.shape[0] * imOrig.shape[1])
        errors.append(mse)
        if flag == 1:
            imYIQ = transformRGB2YIQ(imOrig)
            y, i, q = cv2.split(imYIQ)
            mat = np.asarray([workOn, i, q])
            mat = np.moveaxis(mat, 0, -1)
            pics.append(mat/255)
        else:
            pics.append(workOn/255)

    return pics, errors
