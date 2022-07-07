import cv2 as cv
import numpy as np
import os

input_folder = r'sample/'
result_folder = r'result/'
def main():
    def gray_img(img):  #Convert BGR_image to GRAY_image
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
        return gray

    def threshold(gray_img): #Convert GRAY_image to Threshold_image 
        thresh = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
        return thresh

    def dilation(thresh_img): # Dilate image
        kernel_size = (5,5)
        kernel = np.ones(kernel_size, "uint8")
        mask = cv.dilate(thresh_img, kernel)
        return mask

    def inpaint(mask): # Inpaint image with mask
        result_image = cv.inpaint(image,mask,7,cv.INPAINT_NS)
        return result_image

    for img in  os.listdir(input_folder): #Read image from folder and process
        img_path = os.path.join(input_folder,img)
        image = cv.imread(img_path)
        result_image = inpaint(
                                dilation(
                                            threshold(
                                                        gray_img(image))))
        cv.imwrite(result_folder + str(img).strip('.jpg') + '.jpg', result_image)   #Save result_image to result_folder

if __name__ == '__main__':
    main()

