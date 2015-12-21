import PIL.Image
import numpy as np


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
    def __preprocess(self, net, img):
        return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']

    def __deprocess(self, net):
        img = net.blobs['data'].data[0]
        return np.dstack((img + net.transformer.mean['data'])[::-1])

    def visualize_src(self, net):
        vis = self.__deprocess(net)
        # adjust image contrast and clip
        vis = vis * (255.0 / np.percentile(vis, 99.98))
        vis = np.uint8(np.clip(vis, 0, 255))
        return vis
