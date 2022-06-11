from src.SaliencyDetector import SaliencyDetector
from src.ImageUtils import ImageUtils
from tqdm import tqdm
import os

class MaskGenerator:
    def __init__(self):
        self.detector = SaliencyDetector()

    def __getMaskName(self, image):
        return image[:image.rfind('.')] + '.pbm'

    def generateForImage(self, image):
        if image.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = ImageUtils.loadImage(image)

            mask = self.detector.getSaliency(img)
            mask = ImageUtils.refineMask(mask)
            mask =  ImageUtils.convertArrayToImage(mask)
            mask.save(self.__getMaskName(image))

    def __getImagesList(self, images_root_dir):
        images = []
        for path, subdirs, files in os.walk(images_root_dir):
            for name in files:
                if name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    images.append(os.path.join(path, name))
        return images

    def generateForListOfImages(self, images_root_dir):
        if os.path.isdir(images_root_dir):
            images = self.__getImagesList(images_root_dir)

            for image in tqdm(images, desc = 'Generating masks'):
                self.generateForImage(image)