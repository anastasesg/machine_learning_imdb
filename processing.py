import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from lxml import html

eps = np.finfo(float).eps


class Data:
	"""
	A class to handle the training/test data
	"""
	def __init__(self):
		self.examples = None
		self.labels = None
		self.vocabulary = None
		self.attributes = None
		self.data_frame = None

	def read_vocab(self, filename):
		"""
		A method to read the 'imdb.vocab' file and store it as an np.array
		:param filename: a string containing the vocabulary's filepath
		"""
		if os.path.isfile(filename):
			file = open(filename, 'r', encoding='utf8')
			self.vocabulary = np.array(file.read().splitlines())
			file.close()

	@staticmethod
	def example_open(filepath):
		"""
		A method to quickly open and read a file
		:param filepath: a string containing an example's filepath
		"""
		file = open(filepath, 'r', encoding='utf8')
		string = file.readline()
		file.close()
		string = re.sub(r"[^a-zA-Z0-9-' ]+", '', str(html.fromstring(string.lower()).text_content()))
		return " ".join(set(string.split()))

	def read_examples(self, parent_directory):
		"""
		A method to read every review in the <parent_directory> directory and store it in a pandas data_frame
		:param parent_directory: a string containing the parent directory containing the data
		"""
		if os.path.exists(parent_directory):
			if not os.path.isfile(os.path.join('data', 'examples.csv')):
				examples_paths = [
					os.path.join(parent_directory, os.path.join('train', 'pos')),
					os.path.join(parent_directory, os.path.join('train', 'neg')),
					os.path.join(parent_directory, os.path.join('test', 'pos')),
					os.path.join(parent_directory, os.path.join('test', 'neg'))
				]
				examples, labels = [], []
				for num, path in enumerate(examples_paths):
					string = 'Reading examples {}/4'.format(num + 1)
					file_list, file_length = os.listdir(path), len(os.listdir(path))

					for i, filename in zip(tqdm(range(file_length), desc=string), file_list):
						examples.append(self.example_open(os.path.join(path, filename)))
						labels.append(0 * (path[-3:] == 'neg') + 1 * (path[-3:] == 'pos'))

				self.data_frame = pd.DataFrame({'Examples': examples, 'Category': labels}).sample(frac=1)
				self.data_frame.to_csv(os.path.join('data', 'examples.csv'), index=False)
			else:
				self.data_frame = pd.read_csv(os.path.join('data', 'examples.csv')).sample(frac=1)
			self.examples = np.array(self.data_frame['Examples'].tolist())
			self.labels = np.array(self.data_frame['Category'].tolist())

	def information_gain(self, word):
		"""
		A method to calculate the information gain of a given word
		:param word: a string containing the word whose information gain will be calculated
		:return: a float between 0 and 1
		"""

		X = [int(word in example) for example in self.examples]
		attribute_entropy = 0
		length = len(X)

		for value in (0, 1):
			for category in (0, 1):
				a = sum([int(X[i] == value and self.labels[i] == category) for i in range(length)]) + eps
				b = sum([int(X[i] == value) for i in range(length)]) + eps
				conditional_prob = a / b
				attribute_prob = b / length
				attribute_entropy += - attribute_prob * conditional_prob * np.log2(conditional_prob)
		return 1 - attribute_entropy

	def select_attributes(self, max_length):
		"""
		A method to select attributes for training
		:param max_length: An integer to specify how many attributes to select
		"""
		if os.path.isfile(os.path.join('data', 'vocabulary.csv')):
			self.vocabulary = pd.read_csv(os.path.join('data', 'vocabulary.csv'))['Word'].tolist()
		else:
			info_gains = []
			for num, chunk in enumerate(range(0, len(self.vocabulary), 500)):
				chunked_vocab = self.vocabulary[chunk:chunk+500]
				string = 'Processing chunk {0}/{1}'.format(num + 1, len(self.vocabulary)//500)
				for i, word in zip(tqdm(range(len(chunked_vocab)), desc=string), chunked_vocab):
					info_gains.append(self.information_gain(word))
			print('Writing data to list.')
			tmp_vocab = pd.DataFrame({'Word':self.vocabulary, 'Information Gain': info_gains})
			tmp_vocab.sort_values(by=['Information Gain'], ascending=False)
			tmp_vocab.to_csv(os.path.join('data', 'vocabulary.csv'), index=False)
			self.vocabulary = tmp_vocab['Words'].tolist()
		self.attributes = self.vocabulary[:max_length]

	def load_data(self, max_length, parent_directory='data', vocab_file='./data/imdb.vocab'):
		"""
		A method to load the data, load and process the vocabulary
		:param max_length: an integer to specify how many attributes will be used
		:param parent_directory: a string containing the parent directory containing the data, default 'data'
		:param vocab_file: a string containing the vocabulary's filepath, default 'data/imdb.vocab'
		"""
		self.read_examples(parent_directory)
		self.read_vocab(vocab_file)
		self.select_attributes(max_length)
		self.vectorize()
		return self.examples, self.labels

	@staticmethod
	def data_split(X, y, split):
		"""
		A method to split the data
		:param X: A data frame containing examples
		:param y: A list containing the labels of the examples
		:param split: A float between 0 and 1 to specify split percentage
		:return: 4 numoy arrays
		"""
		split_X1 = np.array(X[:int(len(X) * split)])
		split_X2 = np.array(X[int(len(X) * split):])
		split_y1 = np.array(y[:int(len(y) * split)])
		split_y2 = np.array(y[int(len(y) * split):])
		return split_X1, split_y1, split_X2, split_y2

	def vectorize(self):
		"""
		A method to convert the data to 0-1 arrays
		:param X: A data frame containing examples
		:return: A vectorized example matrix
		"""
		vectorized = []
		for i, example in zip(tqdm(range(len(self.examples)), desc='Vectorizing examples'), self.examples):
			vectorized.append([int(str(word) in example) for word in self.attributes])
		self.examples = np.array(vectorized)
