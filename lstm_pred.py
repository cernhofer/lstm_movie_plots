import pandas as pd 
import numpy as np
import re 
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import collections
import time
import pdb
import pickle
import lstm_train

def get_predictions(input_text):
	# load the data 

	with open ('outfile', 'rb') as fp:
		vocab = pickle.load(fp)
	
	batch_size = 64
	time_steps = 100
	num_batches = 20000

	# *** size of the generated text 
	output_txt_len = 200

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.InteractiveSession(config=config)

	lstm = lstm_train.lstm_rnn(input_size = len(vocab),
					lstm_size = 256, 
					num_layers = 2, 
					output_size = len(vocab), 
					session = sess, 
					eta = 0.003)


	save_model = tf.train.Saver(tf.global_variables())

	save_model.restore(sess, "model.ckpt")


	input_text = input_text.lower()
	# get user input and lower it 

	for i in range(len(input_text)):
		out = lstm.run_step(lstm_train.create_vectors(input_text[i], vocab) , i==0)


	for i in range(output_txt_len):
		element = np.random.choice(range(len(vocab)), p = out ) 
		input_text += vocab[element]

		out = lstm.run_step(lstm_train.create_vectors(vocab[element], vocab) , False )
	
	print(input_text)
	output_string = re.match(r'(.*)\.', input_text)

	print(output_string[0])