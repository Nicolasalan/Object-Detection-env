from PIL import Image
import skimage.morphology
import numpy as np
import cv2

class ImageUtils:
    @staticmethod
    def loadImage(image_path):
        img = Image.open(image_path)
        return ImageUtils.convertImageToArray(img)

    @staticmethod
    def convertImageToArray(image, type = np.uint8):
        return np.array(image).astype(type)

    @staticmethod
    def convertArrayToImage(array):
        return Image.fromarray(array)

    @staticmethod
    def  convertRgbaToRgb(rgba_img):
        return rgba_img[:,:,:3]*np.expand_dims(rgba_img[:,:,3],2)

    @staticmethod
    def convertRgbToRgba(rgb_img, a_channel):
        r_channel, g_channel, b_channel = rgb_img[:, :, 0], rgb_img[:, :, 1], rgb_img[:, :, 2]
        return np.dstack((r_channel, g_channel, b_channel, a_channel))
    
    @staticmethod
    def getMaskContours(mask):
        _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    @staticmethod
    def getLargestContour(contours):
        return max(contours, key = cv2.contourArea)

    @staticmethod
    def getMaskLargestContour(mask):
        contours = ImageUtils.getMaskContours(mask)
        return ImageUtils.getLargestContour(contours) if len(contours) > 0 else contours

    @staticmethod
    def getNoNoiseMask(mask):
        ret, nonoise_mask = cv2.threshold(mask,100,255,cv2.THRESH_TOZERO)
        return cv2.erode(nonoise_mask, np.ones((9, 9), np.uint8))

    @staticmethod
    def getInterestAreaBBox(mask, edge = 10):
        height, width = mask.shape[0], mask.shape[1]
        nonoise_mask = ImageUtils.getNoNoiseMask(mask)
        contour = ImageUtils.getMaskLargestContour(nonoise_mask)
        x,y,w,h = cv2.boundingRect(contour) if len(contour) > 0 else (0,0,width,height)
        return max(0,x-edge), max(0,y-edge), min(x+w+edge,width), min(y+h+edge,height)

    @staticmethod
    def fillInterestArea(recipient_mask, supplier_mask):
        xmin, ymin, xmax, ymax = ImageUtils.getInterestAreaBBox(supplier_mask)
        recipient_mask[ymin:ymax, xmin:xmax] = supplier_mask[ymin:ymax, xmin:xmax]
    
    @staticmethod
    def increaseIntensity(mask, factor):
        mask[mask >  255 - factor] = 255
        mask[mask <= 255 - factor] += factor

    @staticmethod
    def applyOpening(mask, kernel_size, iterations = 1):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations = iterations)

    @staticmethod
    def applyClosing(mask, kernel_size, iterations = 1):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations = iterations)

    @staticmethod
    def applyBlur(mask, sigma):
        return cv2.GaussianBlur(mask, (0,0), sigmaX=sigma, sigmaY=sigma, borderType = cv2.BORDER_DEFAULT)

    @staticmethod
    def fillHoles(mask):
        seed = np.copy(mask)
        seed[1:-1, 1:-1] = mask.max()
        return skimage.morphology.reconstruction(seed, mask, method='erosion').astype(np.uint8)

    @staticmethod
    def getBinaryImage(blur):
        ret, th = cv2.threshold(blur,85,255,cv2.THRESH_BINARY)
        return th

    @staticmethod
    def refineMask(mask):
        final_mask = np.zeros(mask.shape, np.uint8)
        
        ImageUtils.fillInterestArea(final_mask, mask)

        final_mask = ImageUtils.fillHoles(final_mask)
        ImageUtils.increaseIntensity(final_mask, 55)
        final_mask = ImageUtils.applyBlur(final_mask, 3)
        final_mask = ImageUtils.getBinaryImage(final_mask)

        final_mask = ImageUtils.applyClosing(final_mask, (5, 5), 1)
        final_mask = ImageUtils.applyClosing(final_mask, (13, 13), 1)
        final_mask = ImageUtils.applyClosing(final_mask, (21, 21), 1)
        final_mask = ImageUtils.applyClosing(final_mask, (27, 27), 1)

        final_mask = ImageUtils.applyOpening(final_mask, (5, 5), 1)
        final_mask = ImageUtils.applyOpening(final_mask, (13, 13), 1)
        final_mask = ImageUtils.applyOpening(final_mask, (21, 21), 1)
        final_mask = ImageUtils.applyOpening(final_mask, (27, 27), 1)
        
        return final_mask

    @staticmethod
    def cropObjectInImage(img, mask, edge = 10):
        xmin, ymin, xmax, ymax = ImageUtils.getInterestAreaBBox(mask, edge)
        cropped_img = img[ymin:ymax, xmin:xmax]
        cropped_mask = mask[ymin:ymax, xmin:xmax]
        return cropped_img, cropped_mask

