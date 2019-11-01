from peak import *
import matplotlib.pyplot as plt, mpld3
import sys


def walk_forward_validation(data, n_test, cfg):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# fit model
	model = model_fit(train, cfg)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# fit model and make forecast for history
		yhat = model_predict(model, history, cfg)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
  	# estimate prediction error
	error = measure_rmse(test, predictions)
	print(' > %.3f' % error)
	return predictions, test


# define dataset
series = pd.read_csv('long_term_ret.csv', header=0, index_col=0)
data = series.values
# data split
n_test = 30
# cfg
cfg=[12, 100, 200, 150, 12]
true_values=walk_forward_validation(data[:110], n_test, cfg)[1]
predictions=walk_forward_validation(data[:110], n_test, cfg)[0]
plot_it(true_values, predictions)
sys.exit()