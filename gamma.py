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
import cv2
import numpy as np

from ex1_utils import LOAD_GRAY_SCALE

slider_max = 200
window_name = "gamma correction"
pic = cv2.imread("dark.jpg")


def on_trackbar(val):
    global pic
    gamma = val / 100
    corrected = np.power(pic, gamma)
    cv2.imshow(window_name, corrected)


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    global pic
    if rep == 2:
        pic = cv2.imread(img_path, cv2.IMREAD_COLOR)
        pic = np.asarray(pic, np.float)
        pic /= 255
    else:
        pic = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        pic = np.asarray(pic, np.float)
        pic /= 255

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 600, 600)
    cv2.createTrackbar("gamma", window_name, 100, slider_max, on_trackbar)

    on_trackbar(100)
    cv2.waitKey()


def main():
    gammaDisplay('dark.jpg', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
