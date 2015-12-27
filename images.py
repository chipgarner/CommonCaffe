import PIL.Image
import numpy as np
import cv2


class Images:
    # reshape and load image
    def load_image(self, net, image_relative_path):
        img = np.float32(PIL.Image.open(image_relative_path))
        source = net.blobs['data']
        h, w, c = img.shape[:]
        source.reshape(1, 3, h, w)
        source.data[0] = self.__preprocess(net, img)

    def load_image_img(self, net, img):
        source = net.blobs['data']
        h, w, c = img.shape[:]
        source.reshape(1, 3, h, w)
        source.data[0] = self.__preprocess(net, img)

    # a couple of utility functions for converting to and from Caffe's input image layout
    @staticmethod
    def __preprocess(net, img):
        return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']

    @staticmethod
    def __deprocess(net):
        img = net.blobs['data'].data[0]
        return np.dstack((img + net.transformer.mean['data'])[::-1])

    def visualize_src(self, net):
        vis = self.__deprocess(net)
        # adjust image contrast and clip
        vis = vis * (255.0 / np.percentile(vis, 99.98))
        vis = np.uint8(np.clip(vis, 0, 255))
        return vis

    @staticmethod
    def resize_image(height, width, image):
        h, w = image.shape[:2]
        ratio = float(width) / float(height)
        img_in_ratio = float(w) / float(h)

        new_image = np.zeros((height, width, 3), np.uint8)
        if ratio > img_in_ratio:
            new_width = int(float(height) * img_in_ratio)
            img = cv2.resize(image, (new_width, height), interpolation=cv2.INTER_AREA)
            new_image[0:height, (width - new_width) / 2:(width + new_width) / 2] = img
        else:
            new_height = int(float(width) / img_in_ratio)
            img = cv2.resize(image, (width, new_height), interpolation=cv2.INTER_AREA)
            new_image[(height - new_height) / 2:(height + new_height) / 2, 0:width] = img

        return new_image
