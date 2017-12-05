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


class lstm_rnn:
	def __init__(self, 
			 	 input_size, 
			 	 output_size, 
			 	 lstm_size, 
			 	 num_layers, 
			 	 session, 
			 	 eta, 
			 	 name = 'lstm'):

		self.input_size = input_size
		self.output_size = output_size
		self.lstm_size = lstm_size
		self.num_layers = num_layers
		self.session = session
		self.eta = tf.constant(eta)
		self.scope = name

		self.last_output_state = np.zeros((self.num_layers * 2 * self.lstm_size))


		with tf.variable_scope(self.scope):
			self.input_data = tf.placeholder(tf.float32, shape=(None, None, self.input_size), name='input_data')
			self.lstm_init_value = tf.placeholder(tf.float32, shape=(None, self.num_layers * 2 * self.lstm_size), name = 'lstm_init_value')

			self.lstm_cells = [tf.contrib.rnn.BasicLSTMCell(self.lstm_size, forget_bias=1.0, state_is_tuple=False) for i in range(self.num_layers)]
			self.lstm = tf.contrib.rnn.MultiRNNCell(self.lstm_cells, state_is_tuple=False)

			outputs, self.lstm_new_state = tf.nn.dynamic_rnn(self.lstm, self.input_data, initial_state=self.lstm_init_value, dtype=tf.float32)

			self.output_weights = tf.Variable(tf.random_normal((self.lstm_size, self.output_size), stddev=0.01))
			self.output_bias = tf.Variable(tf.random_normal( (self.output_size, ), stddev=0.01 ))

			outputs_reshaped = tf.reshape(outputs, [-1, self.lstm_size])
			network_output = (tf.matmul(outputs_reshaped, self.output_weights) + self.output_bias)

			batch_time_shape = tf.shape(outputs)
			self.final_outputs = tf.reshape(tf.nn.softmax( network_output), (batch_time_shape[0], batch_time_shape[1], self.output_size))

			self.y_batch = tf.placeholder(tf.float32, (None, None, self.output_size))
			y_batch_long = tf.reshape(self.y_batch, [-1, self.output_size])

			self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network_output, labels=y_batch_long))			
			self.train_op = tf.train.RMSPropOptimizer(self.eta, 0.9).minimize(self.cost)


	def run_step(self, x, init_zero_state=True):
		## Reset the initial state of the network.
		if init_zero_state:
			init_value = np.zeros((self.num_layers*2*self.lstm_size,))
		else:
			init_value = self.lstm_last_state

		out, next_lstm_state = self.session.run([self.final_outputs, self.lstm_new_state], feed_dict={self.input_data:[x], self.lstm_init_value:[init_value]   } )

		self.lstm_last_state = next_lstm_state[0]

		return out[0][0]

	def train_batch(self, xbatch, ybatch):
		init_value = np.zeros((xbatch.shape[0], self.num_layers*2*self.lstm_size))

		cost, _ = self.session.run([self.cost, self.train_op], feed_dict={self.input_data:xbatch, self.y_batch:ybatch, self.lstm_init_value:init_value   } )

		return cost

def read_data(fname):

	df = pd.read_csv(fname, encoding='latin1')	
	overview_cat = df.plots.str.cat(sep=' ')
	overview_cat = overview_cat.lower()
	overview_cat = re.sub( r'([a-zA-Z])([,.!])', r'\1 \2', overview_cat)

	return overview_cat

def create_vectors(text_data, vocab):
	data = np.zeros((len(text_data), len(vocab)))

	cnt=0
	for s in text_data:
		v = [0.0]*len(vocab)
		v[vocab.index(s)] = 1.0
		data[cnt, :] = v
		cnt += 1

	return data 

def decode_vectors(array, vocab):
	return vocab[ array.index(1) ]

if __name__ == "__main__":
	# load the data 

	ckpt_file = ""
	training_file = 'data/input_text.csv'
	movies_text = read_data(training_file)

	# get one hot vectors
	pdb.set_trace() 
	vocab = sorted(list(set(movies_text)))

	with open('outfile', 'wb') as fp:
		pickle.dump(vocab, fp)
	
	text_vector = create_vectors(movies_text, vocab)
	
	batch_size = 64
	time_steps = 100

	num_batches = 20000

	# *** size of the generated text 
	output_txt_len = 100

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.InteractiveSession(config=config)

	lstm = lstm_rnn(input_size = len(vocab),
					lstm_size = 256, 
					num_layers = 2, 
					output_size = len(vocab), 
					session = sess, 
					eta = 0.003)


	save_model = tf.train.Saver(tf.global_variables())

	sess.run(tf.global_variables_initializer())


	if ckpt_file == "":
		last_time = time.time()

		batch = np.zeros((batch_size, time_steps, len(vocab)))
		batch_y = np.zeros((batch_size, time_steps, len(vocab)))
		batch_ids = range(text_vector.shape[0] - time_steps - 1)

		for i in range(num_batches):
			batch_id = random.sample(batch_ids, batch_size)

			for j in range(time_steps):
				ind1 = [k+j for k in batch_id]
				ind2 = [k+j+1 for k in batch_id]

				batch[:, j, :] = text_vector[ind1, :]
				batch_y[:, j, :] = text_vector[ind2, :]

			curr_loss = lstm.train_batch(batch, batch_y)

			if i % 100 == 0: 
				new_time = time.time()
				diff = new_time - last_time
				last_time = new_time

				print("batch: ", i, "   loss: ", curr_loss, "   speed: ", (100.0/diff), " batches / s")

		save_model.save(sess, "model.ckpt")


	ckpt_file = "model.ckpt"
	if ckpt_file != "":
		save_model.restore(sess, ckpt_file)

	TEST_PREFIX = str(raw_input('Input:'))
	# get user input and lower it 

	for i in range(len(TEST_PREFIX)):
		out = lstm.run_step(create_vectors(TEST_PREFIX[i], vocab) , i==0)

	gen_str = TEST_PREFIX
	for i in range(output_txt_len):
		element = np.random.choice(range(len(vocab)), p = out ) 
		gen_str += vocab[element]

		out = lstm.run_step(create_vectors(vocab[element], vocab) , False )

	print(gen_str)
	
	output_string = re.match(r'(.*)\.', gen_str)

	print(output_string[0])




























