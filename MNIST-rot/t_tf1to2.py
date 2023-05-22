"""Run MNIST-rot"""

import argparse
import os
import random
import sys
import time
#import urllib2                     #mck
import urllib.request as urllib2    #mck
import zipfile
sys.path.append('../')

import numpy as np
import tensorflow as tf

#import tensorflow.compat.v1 as tf   #mck
#tf.disable_v2_behavior()            #mck

from mnist_model import deep_mnist

def get_weights(filter_shape, W_init=None, std_mult=0.4, name='W') -> tf.Tensor:
	if W_init == None:
		stddev = std_mult*np.sqrt(2.0 / np.prod(filter_shape[:3]))
		W_init = tf.random_normal_initializer(stddev=stddev)

	initv = W_init(shape=filter_shape, dtype=tf.float32)
	weights = tf.Variable(name=name, initial_value=initv, dtype=tf.float32, shape=filter_shape)
	#		initializer=W_init)
	#print(weights)
	return weights

def get_phase_dict(n_in, n_out, max_order, name='b'):
	"""Return a dict of phase offsets"""
	if isinstance(max_order, int):
		orders = range(-max_order, max_order+1)
	else:
		diff = max_order[1]-max_order[0]
		orders = range(-diff, diff+1)

	print(f"get_phase_dict({n_in=}, {n_out=}, {max_order=})")
	print(f"{orders=}")
	phase_dict = {}
	for i in orders:
		init = np.random.rand(1, 1, n_in, n_out) * 2. *np.pi
		init = np.float32(init)
		print(f"{init.shape=}")

		const_init = tf.constant_initializer(init)
		initv = const_init(shape=[1, 1, n_in, n_out], dtype=tf.float32)
		#print(initv.shape)

		phase = tf.Variable(name=name+'_'+str(i), initial_value=initv, dtype=tf.float32,
								shape=[1, 1, n_in, n_out])
			#initializer=tf.constant_initializer(init))
		phase_dict[i] = phase
	return phase_dict


if __name__ == '__main__':
	std_mult = 0.4
	filter_shape = [4, 1, 8]

	weight = get_weights(filter_shape=filter_shape, W_init=None, std_mult=std_mult)

	phase_dict = get_phase_dict(n_in=1, n_out=8, max_order=1)
	print(phase_dict)
	

