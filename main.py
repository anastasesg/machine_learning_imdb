from processing import Data
from logistic_regression import evaluate_model
import getopt, sys


def help():
	help_str = """
Usage:
main.py -m <option> -l <option> -i <option> -s <option> -sk <option> -p <option>

-m:    Specify how many attributes will be used. Strictly an integer. Default = 3000
-l:    Specify the learning rate of the algorithm. Strictly a float. Default = 0.36
-i:    Specify how many times will the gradient decent be repeated. Strictly an integer. Default = 1000
-s:    Specify the steps in the evaluation. Strictly an integer. Default = 1
-k:    Specify if the model from sklearn will be used or not. Strictly a boolean. Default = False
-p:    Specify if a plot will be made or not. Strictly a boolean. Default = False
	"""
	print(help_str)


if __name__ == '__main__':
	argumentList = sys.argv[1:]
	options = "m:l:i:s:k:p:h"

	try:
		attributes, learning_rate, iterations, step, sklearn, plot = 3000, 0.36, 1000, 1, False, False
		arguments, values = getopt.getopt(argumentList, options)
		for argument, value in arguments:
			if argument in ("-h", "--Help"):
				help()
				sys.exit()
			elif argument == "-m":
				attributes = int(value)
			elif argument == "-l":
				learning_rate = float(value)
			elif argument == "-i":
				iterations = int(value)
			elif argument == "-s":
				step = int(value)
			elif argument == "-sk":
				sklearn = bool(value)
			elif argument == "-p":
				plot = bool(value)

		process = Data()
		X, y = process.load_data(attributes, 'data', 'data/imdb.vocab')
		X_test, y_test, X_train, y_train = process.data_split(X=X, y=y, split=0.5)

		stats = evaluate_model(
			X_train,
			y_train,
			X_test,
			y_test,
			learning_rate=learning_rate,
			iterations=iterations,
			step=step,
			sklearn=sklearn,
			to_plot=plot
		)
		print(stats)
	except getopt.error as err:
		print(str(err))
		sys.exit()
