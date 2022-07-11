import cv2 as cv
import numpy as np
import os
import argparse


class Extractor:
    def __init__(self, kernel_size=(5, 5)):
        self.kernel_size = kernel_size

    @staticmethod
    def cvt_bgr_to_gray_img(img: np.ndarray):
        """
        Converting BGR image to Gray image
        :param img : np.ndarray
        :return: gray_img
        """
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return gray_img

    @staticmethod
    def cvt_gray_to_threshold_img(gray_img: np.ndarray):
        """
        Converting Gray image to threshold image
        :param gray_img : np.ndarray
        :return: thresh_img
        """

        # w, h, channel = image.shape

        _, thresh_img = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        return thresh_img

    def dilating_threshold(self, thresh_img: np.ndarray):
        """
        Dilating threshold image
        :param thresh_img : np.ndarray
        :return: mask
        """
        kernel = np.ones(self.kernel_size, "uint8")
        mask = cv.dilate(thresh_img, kernel)
        return mask

    @staticmethod
    def painting_image_with_mask(image: np.ndarray, mask: np.ndarray):
        """
        Painting image with mask
        :param image: np.ndarray
        :param mask: np.ndarray
        :return: result_image
        """
        result_image = cv.inpaint(image, mask, 7, cv.INPAINT_NS)
        return result_image

    def extract(self, image: np.ndarray):
        """
        Extract background from image
        :param image: np.ndarray
        """
        return self.painting_image_with_mask(image,
                                             self.dilating_threshold(
                                                 self.cvt_gray_to_threshold_img(
                                                     self.cvt_bgr_to_gray_img(image))))


def parser_args():
    """
    Initiating argument parser
    :return: args
    """
    parser = argparse.ArgumentParser(description='Input and Output Folder')
    parser.add_argument('--input_folder', type=str, help='Name inputFolder')
    parser.add_argument('--output_folder', type=str, help='Name outputFolder')
    args = parser.parse_args()
    return args


def main(args):
    """
    Running main code
    :param args
    """
    input_folder = args.input_folder
    result_folder = args.output_folder
    extraction = Extractor()
    for img in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img)
        image = cv.imread(img_path)
        result_image = extraction.extract(image)
        cv.imwrite(os.path.join(result_folder, img), result_image)


if __name__ == '__main__':
    arg = parser_args()
    main(arg)
