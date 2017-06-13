import os, sys
os.environ['GLOG_minloglevel'] = '2'    # suprress Caffe verbose prints
import caffe
import settings
import numpy as np
from numpy.linalg import norm
import scipy.misc, scipy.io
import argparse 
import util
from sampler import Sampler
from sampling_class import ClassConditionalSampler
from sampling_class import get_code
import shutil


def conditional_sampling_from_real_image(encoder_definition, encoder_weights,
					 generator_definition, generator_weights,
                                         net_definition, net_weights,
                                         units, xy,
                                         n_iters,
                                         reset_every,
                                         save_every,
                                         threshold,
                                         epsilon1, epsilon2, epsilon3, epsilon4,
                                         lr, lr_end,
                                         seed,
                                         opt_layer,
                                         act_layer,
                                         init_file,
                                         output_dir,
                                         write_labels=False):
    try:
        os.stat(output_dir)
        shutil.rmtree(output_dir)
        os.mkdir(output_dir)
        os.mkdir(output_dir + "/samples")
    except:
        os.mkdir(output_dir)
        os.mkdir(output_dir + "/samples")

    # encoder and generator for images 
    encoder = caffe.Net(encoder_definition, encoder_weights, caffe.TEST)
    generator = caffe.Net(generator_definition, generator_weights, caffe.TEST)

    # condition network, here an image classification net
    net = caffe.Classifier(net_definition, net_weights,
                               mean = np.float32([104.0, 117.0, 123.0]), # ImageNet mean
                               channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB

    # Fix the seed
    np.random.seed(seed)

    # Sampler for class-conditional generation
    sampler = ClassConditionalSampler()
    inpainting = None

    if init_file != "None":

        # Pre-compute masks if we want to perform inpainting 
        if epsilon4 > 0:
            mask, neg = util.get_mask()
        else:
            neg = None

        # Get the code for the masked image
        start_code, start_image = get_code(encoder=encoder, path=init_file,
            layer=opt_layer, mask=neg)

        # Package settings for in-painting experiments
        if epsilon4 > 0:
            inpainting = {
                "mask"      : mask,
                "mask_neg"  : neg,
                "image"     : start_image,
                "epsilon4"  : epsilon4
            }

        print "Loaded init code: ", start_code.shape
    else:
        # shape of the code being optimized
        shape = generator.blobs[settings.generator_in_layer].data.shape
        start_code = np.random.normal(0, 1, shape)
        print ">>", np.min(start_code), np.max(start_code)

    # Separate the dash-separated list of units into numbers
    conditions = [ { "unit": int(u), "xy": xy } for u in units.split("_") ]       

    # Optimize a code via gradient ascent
    output_image, list_samples = sampler.sampling(condition_net=net,
        image_encoder=encoder, image_generator=generator, 
        gen_in_layer=settings.generator_in_layer,
        gen_out_layer=settings.generator_out_layer, start_code=start_code, 
        n_iters=n_iters, lr=lr, lr_end=lr_end, threshold=threshold, 
        layer=act_layer, conditions=conditions,
        epsilon1=epsilon1, epsilon2=epsilon2, epsilon3=epsilon3,
        inpainting=inpainting,
        output_dir=output_dir, 
        reset_every=reset_every, save_every=save_every)

    # Output image
    filename = "%s/%s_%04d_%04d_%s_h_%s_%s_%s_%s__%s.jpg" % (
            output_dir,
            act_layer, 
            conditions[0]["unit"],
            n_iters,
            lr,
            str(epsilon1),
            str(epsilon2),
            str(epsilon3),
            str(epsilon4),
            seed
        )

    if inpainting != None:
        output_image = util.stitch(start_image, output_image) 

    # Save the final image
    util.save_image(output_image, filename)
    print "%s/%s" % (os.getcwd(), filename)

    # Write labels to images
    print "Saving images..."
    for p in list_samples:
        img, name, label = p
        util.save_image(img, name)
        if write_labels:
            util.write_label_to_img(name, label)
    return output_image, list_samples
