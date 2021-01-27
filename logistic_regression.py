import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression


def plot(x, y, legends, labels, title):
	plt.figure()
	plt.title(title)
	for i in range(len(y)):
		plt.plot(x, y[i], label=legends[i])
	plt.xlabel(labels[0])
	plt.ylabel(labels[1])
	plt.legend()


def confusion_matrix(y, prediction):
	return np.array(
		[
			[
				sum([int(int(prediction[i]) == 1 and y[i] == 1) for i in range(len(y))]),
				sum([int(int(prediction[i]) == 1 and y[i] == 0) for i in range(len(y))])
			],
			[
				sum([int(int(prediction[i]) == 0 and y[i] == 1) for i in range(len(y))]),
				sum([int(int(prediction[i]) == 0 and y[i] == 0) for i in range(len(y))])
			]
		]
	)


def evaluate_model(x_train, y_train, x, y, learning_rate, iterations, step, sklearn, to_plot):
	percentage = []

	train_acc, train_pre, train_rec, train_F1s = [], [], [], []
	test_acc, test_pre, test_rec, test_F1s = [], [], [], []

	steps = range(step, 100 + step, step)
	for i, percent in zip(tqdm(range(len(steps)), desc='Evaluating model'), steps):
		x_percent, y_percent = (int(percent * len(x_train)) // 100), (int(percent * len(y_train)) // 100)
		cropped_X, cropped_Y = x_train[:x_percent], y_train[:y_percent]

		if sklearn:
			model = LogisticRegression(solver='liblinear', random_state=0)
			model.fit(cropped_X, cropped_Y)

			train_y_pred = model.predict(x_train)
			test_y_pred = model.predict(x)
		else:
			model = logistic_regression(learning_rate, iterations, False)
			model.fit(cropped_X, cropped_Y)

			train_y_pred = model.predict(x_train)
			test_y_pred = model.predict(x)

		train_confusion_matrix = confusion_matrix(y_train, train_y_pred)
		test_confusion_matrix = confusion_matrix(y, test_y_pred)

		percentage.append(percent)

		train_acc.append(sum(np.diagonal(train_confusion_matrix)) / train_confusion_matrix.sum())
		train_pre.append(train_confusion_matrix[0][0] / train_confusion_matrix.sum(axis=1)[0])
		train_rec.append(train_confusion_matrix[0][0] / train_confusion_matrix.sum(axis=0)[0])
		train_F1s.append(2 * train_pre[i] * train_rec[i] / (train_pre[i] + train_rec[i]))

		test_acc.append(sum(np.diagonal(test_confusion_matrix)) / test_confusion_matrix.sum())
		test_pre.append(test_confusion_matrix[0][0] / test_confusion_matrix.sum(axis=1)[0])
		test_rec.append(test_confusion_matrix[0][0] / test_confusion_matrix.sum(axis=0)[0])
		test_F1s.append(2 * test_pre[i] * test_rec[i] / (test_pre[i] + test_rec[i]))

	__tmp = {
		'Train Accuracy': train_acc,
		'Train Precision': train_pre,
		'Train Recall': train_rec,
		'Train F1 Score': train_F1s,
		'Test Accuracy': test_acc,
		'Test Precision': test_pre,
		'Test Recall': test_rec,
		'Test F1 Score': test_F1s,
		'Train Percent': percentage
	}
	stats = pd.DataFrame(__tmp)
	stats.to_csv('statistics.csv', index=False)

	if to_plot:
		plot(percentage, [train_acc, test_acc], ['Training', 'Test'], ['%', 'Accuracy'], 'Accuracy Diagram')
		plot(percentage, [train_pre, test_pre], ['Training', 'Test'], ['%', 'Precision'], 'Precision Diagram')
		plot(percentage, [train_rec, test_rec], ['Training', 'Test'], ['%', 'Recall'], 'Recall Diagram')
		plot(percentage, [train_F1s, test_F1s], ['Training', 'Test'], ['%', 'F1 Score'], 'F1 Score Diagram')

	return stats


class logistic_regression:
	"""
	A class to initialize, train and predict categories with the logistic regression method
	"""

	def __init__(self, learning_rate, iterations, verbose=False):
		"""
		Initialization of the class
		:param learning_rate: Float. Specify the learning rate of the algorithm
		:param iterations: Integer. Specify how many times will the gradient decent be repeated
		"""
		self.weight = None
		self.learning_rate = learning_rate
		self.iterations = iterations
		self.verbose = verbose

	def __intercept(self, X):
		intercept = np.ones((X.shape[0], 1))
		return np.concatenate((intercept, X), axis=1)

	def __sigmoid(self, z):
		return 1 / (1 + np.exp(-z))

	def __loss(self, h, y):
		return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

	def fit(self, X, y):
		X = self.__intercept(X)
		self.weight = np.zeros(X.shape[1])

		if self.verbose:
			for i in tqdm(range(self.iterations), desc='Fitting data'):
				z = np.dot(X, self.weight)
				prediction = self.__sigmoid(z)
				gradient = np.dot(X.T, (prediction - y)) / y.size
				self.weight -= self.learning_rate * gradient
		else:
			for i in range(self.iterations):
				z = np.dot(X, self.weight)
				prediction = self.__sigmoid(z)
				gradient = np.dot(X.T, (prediction - y)) / y.size
				self.weight -= self.learning_rate * gradient

	def predict_probability(self, X):
		X = self.__intercept(X)
		return self.__sigmoid(np.dot(X, self.weight))

	def predict(self, X):
		return self.predict_probability(X) >= 0.5
