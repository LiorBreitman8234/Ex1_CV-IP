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
        if im is None:
            print("image not found")
            return np.zeros(255)
        mat = np.asarray(im, np.float)
        mat = mat / 255
    else:
        im = cv2.imread(filename, cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if im is None:
            print("image not found")
            return np.zeros(255)
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
    if im == np.zeros(255):
        print("error loading image")
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
    pic = imOrig
    if len(imOrig.shape) == 3:
        y, i, q = cv2.split(transformRGB2YIQ(imOrig))
        pic = y
    norm_im = cv2.normalize(pic, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_im = norm_im.astype(int)
    origHist = np.histogram(norm_im, bins=256)
    cumSum = np.cumsum(origHist[0])
    norm_cumSum = (cumSum / cumSum.max())
    LUT = np.ceil(norm_cumSum * 255)
    new_im = numpy.copy(pic)
    for i in range(256):
        new_im[norm_im == i] = LUT[i]
    histNew = np.histogram(new_im, bins=256)

    if len(imOrig.shape) == 3:
        y, i, q = cv2.split(transformRGB2YIQ(imOrig))
        new_im = new_im / 255
        new_pic = np.dstack((new_im, i, q))
        new_pic = transformYIQ2RGB(new_pic)
        return new_pic, origHist[0], histNew[0]
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
    if imOrig is None:
        print("error loading image")
        return [], []
    # if grayscale, run the algorithm
    if len(imOrig.shape) == 2:
        return QuantProcess(imOrig, nQuant, nIter)

    # else take the y of the yiq transformation and use it
    imYIQ = transformRGB2YIQ(imOrig)
    y, i, q = cv2.split(imYIQ)
    pics, errors = QuantProcess(y.copy(), nQuant, nIter)
    fixedPics = []
    for pic in pics:
        fixed_pic = np.dstack((pic, i, q))
        fixed_pic = transformYIQ2RGB(fixed_pic)
        fixedPics.append(fixed_pic)

    return fixedPics, errors


def QuantProcess(pic, nQuant, nIter):
    """
    this function is the Quantization process itself
    :param pic: the picture to run the algorithm on
    :param nQuant: the amount of colors wanted
    :param nIter: how many iteration to do
    :return: list of the images and list of errors in each stage
    """
    # normalizing picture and using histogram
    norm_im = cv2.normalize(pic, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_im = norm_im.astype(int)
    im_flat = norm_im.ravel().astype(int)
    hist = np.histogram(im_flat, bins=256)

    # setting up initial boundaries
    amountOfPixelsInRange = (pic.shape[0] * pic.shape[1]) / nQuant
    count = 0
    bounds = [0]
    for i in range(256):
        if count < amountOfPixelsInRange:
            count += hist[0][i]
        else:
            count = 0
            bounds.append(i)
    bounds.append(255)
    bounds = np.asarray(bounds, dtype=int)

    pics = []
    errors = []

    # looping for the amount of iterations required
    for j in range(nIter):
        means = []
        # calculating the mean for each part of the histogram
        for i in range(nQuant):
            array = hist[0][bounds[i]:bounds[i + 1]]
            z_i = np.average(array, weights=range(len(array)))
            idx = (np.abs(array - z_i)).argmin() + bounds[i]
            means.append(idx)

        # setting the new pic with the mean values
        Quant_pic = np.zeros_like(norm_im)
        for i in range(len(bounds) - 1):
            Quant_pic[norm_im > bounds[i]] = means[i]

        # calculating mse
        mse = np.sqrt((norm_im - Quant_pic) ** 2).mean()
        errors.append(mse)

        Quant_pic = Quant_pic / 255
        pics.append(Quant_pic)

        # setting new boundaries
        runOver = np.arange(1, len(means))
        for i in runOver:
            bounds[i] = np.floor((means[i - 1] + means[i]) / 2)

    return pics, errors
