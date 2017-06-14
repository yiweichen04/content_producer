from __future__ import print_function
import sys
sys.path.insert(0, 'src')
import transform, numpy as np, vgg, pdb, os
import scipy.misc
import tensorflow as tf
from utils import save_img, get_img, exists, list_files
from argparse import ArgumentParser
from collections import defaultdict
import time
import subprocess
import numpy

def style(image, checkpoint_dir):
    g = tf.Graph()
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    with g.as_default(), tf.Session(config=config) as sess:
        batch_shape = (1,) + image.shape
        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape,
                                         name='img_placeholder')
        preds = transform.net(img_placeholder)
        saver = tf.train.Saver()
        if os.path.isdir(checkpoint_dir):
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found...")
        else:
            saver.restore(sess, checkpoint_dir)

        result = sess.run(preds, feed_dict={img_placeholder: image[np.newaxis,...]})
        result = np.clip(result[0,...], 0, 255).astype(np.uint8)
        return result

# Unit test
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # Style an image
    image = scipy.misc.imread('blue_bird.jpg', mode='RGB')
    styled_image = style(image, 'models/udnie')
    # Plot out result
    plt.figure()
    plt.subplot(1,2,1)
    plt.axis('off')
    plt.imshow(image)
    plt.subplot(1,2,2)
    plt.axis('off')
    plt.imshow(styled_image)
    plt.show()
