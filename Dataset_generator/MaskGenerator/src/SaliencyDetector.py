from src.ImageUtils import ImageUtils
import tensorflow as tf
import numpy as np
import os

class SaliencyDetector:
    def __init__(self):
        self.checkpoint = os.path.join(os.path.dirname( __file__ ), '../model/salience_model')  
        self.meta_graph = os.path.join(os.path.dirname( __file__ ), '../model/meta_graph/my-model.meta')
        self.g_mean = np.array(([126.88,120.24,112.19])).reshape([1,1,3])
        self.gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction = 1.0)
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options = self.gpu_options))

        self.__initialize()
        
        self.image_batch = tf.compat.v1.get_collection('image_batch')[0]
        self.pred_mattes = tf.compat.v1.get_collection('mask')[0]

    def getSaliency(self, img):
        height, width = img.shape[0], img.shape[1]

        img = self.__prepareImage(img)

        saliency = self.__detect(img)
        saliency = self.__normalizesMap(saliency)

        saliency = ImageUtils.convertArrayToImage(saliency).resize((width, height))
        return ImageUtils.convertImageToArray(saliency)

    def __prepareImage(self, img):
        img = self.__removeAlphaChannel(img)
        return self.__resizeForModel(img)

    def __removeAlphaChannel(self, img):
        if img.shape[2] == 4:
            return ImageUtils.convertRgbaToRgb(img)
        else:
            return img

    def __resizeForModel(self, img):
        img = ImageUtils.convertArrayToImage(img).resize((320, 320))
        img = ImageUtils.convertImageToArray(img, np.float32)
        return np.expand_dims(img - self.g_mean, 0)

    def __detect(self, img):
        return np.squeeze(self.sess.run(self.pred_mattes,feed_dict = {self.image_batch:img}))

    def __normalizesMap(self, saliency_map):
        return ((saliency_map / saliency_map.max())  * 255.999)

    def __initialize(self):
        print("Initializing TensorFlow session!")
        tf.compat.v1.disable_eager_execution()

        saver = tf.compat.v1.train.import_meta_graph(self.meta_graph)
        saver.restore(self.sess, tf.train.latest_checkpoint(self.checkpoint))

    def __exit__(self):
        print("Closing TensorFlow session!")
        self.sess.close()



