'''This script trains a word embedding using Word2vec from gensim
I used mainly used tflearn to train a 1 dimensional CNN, with three
different sizes of filters (3,4,5), the three outputs are then
merged into a one dimensional vector and  as in  Kim Yoon (Convolutional Neural Networks for Sentence Classification).
Takes around 1 hour for 10 epochs in my machine....
GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
'''

# -*- coding: utf-8 -*-
import tensorflow as tf
import keras
import tflearn
from keras.preprocessing.text import Tokenizer
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical, pad_sequences, VocabularyProcessor
from tensorflow.contrib import learn
from gensim.models import Word2Vec
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter, defaultdict
import re
from os import listdir
import numpy as np
import glob
import os
import random
import string 




# Number of epochs for training the CNNs
NB_EPOCHS = 4
MAX_LENGHT = 66




def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_preprocesses():
	'''
	Loads and cleans the data
	'''
	#Load labels
	with open("ytrain.txt") as f:
		content = f.readlines()
	labels = [x.strip() for x in content]
	
	with open("xtrain.txt") as f:
		content = f.readlines()
	sentences = [x.strip() for x in content]

	with open("xtest.txt") as f:
		content = f.readlines()
	test_sentences  = [x.strip() for x in content]


	new_s = [clean_str(x) for x in sentences]
	test_sentences = [clean_str(x) for x in test_sentences]

	return new_s, labels, test_sentences


def lstm(trainX, trainY, valX, valY ,testX ,input_weights):
	'''
	Standard lstm Network.
	'''
	# Network building LSTM
	net = tflearn.input_data([None, MAX_LENGHT])
	net = tflearn.embedding(net, input_dim=input_weights.shape[0], output_dim=input_weights.shape[1], trainable=True, name="EmbeddingLayer")
	net = tflearn.lstm(net, 128, dropout=0.5)
	net = tflearn.fully_connected(net,12, activation='softmax')
	net = tflearn.regression(net, optimizer='adam', learning_rate=0.01,
	                         loss='categorical_crossentropy')
	# Training
	model = tflearn.DNN(net, tensorboard_verbose=0)

	# Add embedding weights into the embedding layer
	embeddingWeights = tflearn.get_layer_variables_by_name("EmbeddingLayer")[0]
	model.set_weights(embeddingWeights, input_weights)

	model.fit(trainX, trainY, n_epoch = NB_EPOCHS, validation_set=(valX, valY), show_metric=True,
	         batch_size=32)
	y_result = model.predict(testX)
	return y_result



def cnn_3_filters(trainX, trainY, valX, valY ,testX ,input_weights):
	'''
	A CNN with three convolutional layers as in Kim Yoon (Convolutional Neural Networks for Sentence Classification)

	'''
	# Building convolutional network
	network = input_data(shape=[None, MAX_LENGHT], name='input')
	network = tflearn.embedding(network, input_dim=input_weights.shape[0], output_dim=input_weights.shape[1], trainable=True, name="EmbeddingLayer")
	branch1 = conv_1d(network, 128, 3, padding='valid', activation='relu', regularizer="L2")
	branch2 = conv_1d(network, 128, 4, padding='valid', activation='relu', regularizer="L2")
	branch3 = conv_1d(network, 128, 5, padding='valid', activation='relu', regularizer="L2")
	network = merge([branch1, branch2, branch3], mode='concat', axis=1)
	network = tf.expand_dims(network, 2)
	network = global_max_pool(network)
	network = dropout(network, 0.5)
	network = fully_connected(network,12, activation='softmax')
	network = regression(network, optimizer='adam', learning_rate=0.001, 
	                     loss='categorical_crossentropy', name='target')
	# Training
	model = tflearn.DNN(network, tensorboard_verbose=1)


	# Add embedding weights into the embedding layer
	embeddingWeights = tflearn.get_layer_variables_by_name("EmbeddingLayer")[0]
	model.set_weights(embeddingWeights, input_weights)


	print("Start trianing CNN...")
	model.fit(trainX, trainY, n_epoch = NB_EPOCHS, validation_set=(valX, valY), shuffle=True, show_metric=True, batch_size=32)
	
	y_result = model.predict(testX)
	return y_result



if __name__ == '__main__':
	## Load Glove vectors. Only tested on 50 dimension vectors
	glove_dict = {}
	globe_data = 'glove/glove.6B/glove.6B.50d.txt'
	f = open(globe_data)
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		glove_dict[word]= coefs
	f.close()


	#Load train data 
	train_sentences, labels, test_sentences = load_preprocesses()

	# Append train and test sentences
	all_text = train_sentences + test_sentences

	#Find max length in a sentence
	MAX_SEQUENCE_LENGTH = max([len(x.split(" ")) for x in all_text])
	EMBEDDING_DIM = len(glove_dict['the'])
	print('Max lenght in sentence %s . '  % MAX_SEQUENCE_LENGTH)
	

	#Convert sentences to tokens, trained on all test
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(all_text)

	# Prepare train data
	sequences = tokenizer.texts_to_sequences(train_sentences)
	X_train = tflearn.data_utils.pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, value =0.)  # Add padding
	
	#One-hot vector to labels
	y_train = to_categorical(labels, nb_classes=12)

	#Prepare test data
	sequences_test = tokenizer.texts_to_sequences(test_sentences)
	X_test = tflearn.data_utils.pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH, value =0.)  # Add padding

	# Word to id dictionary
	word_index = tokenizer.word_index 
	unique_words = len(word_index)
	print('Found %s unique words. '  % unique_words)
	print('Shape of data tensor:', X_train.shape)
	print('Shape of label tensor:', y_train.shape)
	

	#Create embedding matrix of GloVe vectors
	print("Create GloVe Embedding....")
	embedding_Glove = np.zeros((unique_words + 1, EMBEDDING_DIM))
	for word, index in word_index.items():
		vector = glove_dict.get(word)
		if vector is not None:
			embedding_Glove[index] = vector
	
	#Train Word2vec model. Model is trained using all text ( train and test)
	print("Train Word2Vec model...")
	X = [keras.preprocessing.text.text_to_word_sequence(x) for x in all_text]
	model_w2v = Word2Vec(X, size=EMBEDDING_DIM, window=5, min_count=0)
	
	#Create embedding matrix Word2Vec
	embedding_word2vec = np.zeros((unique_words + 1, EMBEDDING_DIM))
	for word, i in word_index.items():
		embedding_word2vec[i] = model_w2v[word]

	#Split data into train and validation data
	X_train, X_val, Y_train, Y_val = train_test_split(X_train, y_train, test_size =0.1, random_state = 0)


	#Train using all train data. 
	predict = lstm(X_train, Y_train, X_val, Y_val, X_test, embedding_word2vec)
	#predict = cnn_3_filters(X_train, Y_train, X_val, Y_val, X_test, embedding_word2vec)
	#predict = cnn_3_filters(X_train, Y_train, X_val, Y_val, X_test, embedding_Glove)

	# Map one hot vectors to real predictions [0-11]
	output_labels = np.zeros(len(predict))
	for i in range(len(predict)):
		output_labels[i] = predict[i].index(max(predict[i]))

	#Save predictions to output file
	np.savetxt("ytest.txt", output_labels.astype(int), fmt='%i', newline='\n')
