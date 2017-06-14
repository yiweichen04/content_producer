# import library
import os
import scipy.misc
import numpy as np
import tensorflow as tf

from model import DCGAN
from utils import pp, visualize, to_json, test, get_image, save_images, doresize2, merge_overlap

# parameter setting
flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 128, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_string("dataset", "place", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS


def super_resolution_initialize():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	#  sess = tf.Session()
	# initialization
	dcgan = DCGAN(sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size,
				dataset_name=FLAGS.dataset, is_crop=FLAGS.is_crop, checkpoint_dir=FLAGS.checkpoint_dir)
	# load model
	print ('Load model')
	dcgan.load_with_name(FLAGS.checkpoint_dir, 'DCGAN.model-3002')
	
	return sess, dcgan

def super_resolution(sess, dcgan, img_path):
	# test 
	#with tf.Session() as sess:
	## initialization
	#	dcgan = DCGAN(sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size,
	#				dataset_name=FLAGS.dataset, is_crop=FLAGS.is_crop, checkpoint_dir=FLAGS.checkpoint_dir)
	#	# load model
	#	print ('Load model')
	#	dcgan.load_with_name(FLAGS.checkpoint_dir, 'DCGAN.model-3002')
	#	#img_path = './input/test.jpg'
        img_name = img_path.split("/")[-1]
	img = get_image(img_path, 200, is_crop=False)  # read image
	img = doresize2(img, [200,200])  # resize image to 200x200
	img_re = scipy.misc.imresize(img, [768,768])
	scipy.misc.imsave('../Result/sr/LR/' + img_name,img_re)
	
	
	# divide image
	sample = [img[24*i:24*i+32,24*j:24*j+32,:] for i in range(8) for j in range(8)]
	batch_inputs = np.array(sample).astype(np.float32)
	
	# SR process
	samples = sess.run(dcgan.G, feed_dict={dcgan.inputs: batch_inputs})
	save_img = merge_overlap(samples, [8, 8], [24, 24])
	scipy.misc.imsave('../Result/sr/SR/' + img_name,save_img)	
	
	print ('Finish!')
	return  np.asarray(save_img)
