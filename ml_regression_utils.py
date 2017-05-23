# coding=utf-8
__author__ = 'nadyaK'
__date__ = '04/09/2017'

from math import sqrt
import ml_numpy_utils as np_utils
import ml_graphlab_utils as gp
import matplotlib.pyplot as plt

class SimpleLinearRegression(object):
	"""simple linear regression model """

	def __init__(self):
		self.__name__ = "simple_linear_regression_model"

	def simple_linear_regression(self, input_feature, output):
		"""compute the slope(w1) & intercept(w0) for a simple linear regression """

		# compute the sum of input_feature and output
		sum_yi = (output).sum()
		sum_xi = (input_feature).sum()

		# compute the product of the output and the input_feature and its sum
		sum_xiyi = (output * input_feature).sum()

		# compute the squared value of the input_feature and its sum
		sum_xi2 = (input_feature ** 2).sum()
		N = float(len(input_feature))
		numerator = sum_xiyi - ((sum_yi * sum_xi) / N)

		# use the formula for the slope (w1)
		slope = (numerator) / float(sum_xi2 - ((sum_xi * sum_xi) / N))

		# use the formula for the intercept (w0)
		intercept = sum_yi / N - slope * (sum_xi / N)

		return (intercept,slope)

	def inverse_regression_predictions(self, output, intercept, slope):
		""" yi_predict = (w0 + w1*xi)
		   inverse:   xi = (yi_predict - w0)/w1 """
		return ((output - intercept) / float(slope))

	def get_residual_sum_of_squares(self, input_feature, output, intercept, slope):
		"""make predictions and evaluate the model using Residual Sum of Squares (RSS)"""
		predicted_values = get_regression_predictions(input_feature,intercept,slope)
		return compute_RSS(predicted_values,output)

class GrandientDescent(object):
	"""Gradient Descent model """

	def __init__(self):
		self.__name__ = "gradient_descent_model"
		self.initial_intercept = 0
		self.initial_slope = 0

	#==================================================================
	#                Regression Grandient Descent Model
	#==================================================================
	def regression_gradient_descent(self, feature_matrix,output,initial_weights,step_size,tolerance):
		converged = False
		weights = np_utils.np.array(initial_weights) # make sure it's a numpy array
		while not converged:
			# compute the predictions based on feature_matrix and weights using your predict_output() function
			predictions = np_utils.predict_output(feature_matrix,weights)
			# compute the errors as predictions - output
			errors = predictions - output
			gradient_sum_squares = 0 # initialize the gradient sum of squares
			# while we haven't reached the tolerance yet, update each feature's weight
			for i in range(len(weights)): # loop over each weight
				# Recall that feature_matrix[:, i] is the feature column associated with weights[i]
				# compute the derivative for weight[i]:
				feature = feature_matrix[:,i]
				derivative = 2 * np_utils.feature_derivative(errors,feature)
				# add the squared value of the derivative to the gradient sum of squares (for assessing convergence)
				gradient_sum_squares += derivative ** 2
				# subtract the step size times the derivative from the current weight
				weights[i] -= step_size * derivative
			# compute the square-root of the gradient sum of squares to get the gradient magnitude:
			gradient_magnitude = sqrt(gradient_sum_squares)
			if gradient_magnitude < tolerance:
				converged = True
		return (weights)

	def compute_prediction_errors(self, y_real, y_predicted):
		return (y_predicted - y_real)

	def update_intercept(self, derivate_slope, step_size, intercept):
		"""compute the derivative: sum(errors)"""
		adjustment = step_size * derivate_slope
		new_intercept = intercept - adjustment
		return (new_intercept)

	def update_slope(self, derivate, step_size, slope):
		"""compute the derivative: sum(errors*input)"""
		adjustment = step_size * derivate
		new_slope = slope - adjustment
		return new_slope

	def calculate_magnitude(self, derivate_intercept, derivate_slope):
		return sqrt(derivate_intercept**2 + derivate_slope**2)

	#==================================================================
	#                Grandient Descent Model
	#==================================================================
	def calculate_gradiendt_descent(self, x, y, step_size, tolerance):
		""" 1. Compute the predicted values given the current slope and intercept
			2. Compute the prediction errors (prediction - Y)
			3. Update the intercept:
				compute the derivative: sum(errors)
				compute the adjustment as step_size times the derivative
				decrease the intercept by the adjustment
			4. Update the slope:
				compute the derivative: sum(errors*input)
				compute the adjustment as step_size times the derivative
				decrease the slope by the adjustment
			5. Compute the magnitude of the gradient
			6. Check for convergence
		"""
		converged = False
		step = 1

		while not converged:
			# print "\n\nstep: %s"% step
			y_predict = get_regression_predictions(x,self.initial_intercept,self.initial_slope)
			# print "1) predictions: %s" % y_predict
			errors = self.compute_prediction_errors(y, y_predict)
			# print "2) predictions: %s" % errors
			derivate_intercept = sum(errors)
			self.initial_intercept = self.update_intercept(derivate_intercept, step_size, self.initial_intercept)
			# print "3) New Intercept: %s" % self.initial_intercept
			derivate_slope = sum(errors*x)
			self.initial_slope = self.update_slope(derivate_slope, step_size, self.initial_slope)
			# print "4) New Slope: %s" %self.initial_slope
			magnitude = self.calculate_magnitude(derivate_intercept, derivate_slope)
			# print "5) Magnitude: %s" %magnitude
			check_if_converged  = not(magnitude > tolerance)
			# print "6) Converge?: %s" % check_if_converged
			if check_if_converged:
				converged = True

			# if step in [1,2,3,78]:
			# 	predicted_y = self.initial_intercept + self.initial_slope*x
			# 	plt.plot(x,y,'r.', x,predicted_y,'g-', [x.mean()], [y.mean()],'b*')
			# 	plt.ylim(0,23)
			# 	# plt.show()
			# 	plt.savefig(os.path.join(self.save_plots, 'step_%s.png'%step))
			step+=1
		return self.initial_intercept, self.initial_slope

class RidgeRegression(object):
	"""Gradient Descent model """

	def __init__(self):
		self.__name__ = "ridge_regression_model"

	def ridge_regression_gradient_descent(self, feature_matrix,output,initial_weights,
											step_size,l2_penalty,max_iterations=100, debug=True):

		weights = np_utils.np.array(initial_weights)
		iteration = 0 # iteration counter
		print_frequency = 1  # for adjusting frequency of debugging output

		#while not reached maximum number of iterations:
		while iteration < max_iterations:
			iteration += 1  # increment iteration counter
			if (iteration in [10, 100]) and debug:
				print_frequency = iteration

			# compute the predictions(dot-product) between feature_matrix and weights
			predictions = np_utils.predict_output(feature_matrix,weights)

			# compute the errors as predictions - output
			errors = predictions - output

			# from time to time, print the value of the cost function
			if (iteration % print_frequency == 0) and debug:
				print('Iteration: %s' % iteration)
				print('Cost function: %s'% np_utils.compute_cost_function(errors, weights, l2_penalty))

			for i in xrange(len(weights)): # loop over each weight
				# Recall that feature_matrix[:,i] is the feature column associated with weights[i]
				# compute the derivative for weight[i].
				#(Remember: when i=0, you are computing the derivative of the constant!)
				is_constant = True if i == 0 else False
				feature = feature_matrix[:,i]
				derivative = np_utils.feature_derivative_ridge(errors,feature,weights[i],l2_penalty,is_constant)
				# subtract the step size times the derivative from the current weight
				weights[i] -= step_size * derivative

		return iteration, weights

class LassoRegression(object):
	"""Lasso Regression Model"""

	def __init__(self):
		self.__name__ = 'lasso_regression_model'

	def create_lasso_regression(self, training, validation, model_info, max_nonzeros=0):
		""" customized function to answer week5-quiz"""
		RSS_best, L1_best, best_model = None, None, None
		l1_penalty_min, l1_penalty_max = None, None
		max_num_of_nnz, min_num_of_nnz = None, None
		continue_search = True
		L1_penalities = model_info['L1_penalties']
		target = model_info['target']#price
		features = model_info['features']

		for l1_penalty in L1_penalities:
			current_model = gp.graphlab.linear_regression.create(training,target=target,features=features,
				validation_set=None,verbose=False,l2_penalty=0.,l1_penalty=l1_penalty)
			predictions = current_model.predict(validation)

			RSS = compute_RSS(predictions,validation['price'])
			print "L1 penalty (%.2f)\t\tRSS=%s" % (l1_penalty,RSS)
			if RSS_best is None or RSS < RSS_best:
				RSS_best = RSS
				L1_best = l1_penalty
				best_model = current_model

			if max_nonzeros:
				current_num_nnz = current_model.coefficients['value'].nnz()
				print "\t\tNon-Zeros: %s" % current_num_nnz

				if continue_search:
					#The largest l1_penalty that has more non-zeros than max_nonzeros
					if current_num_nnz > max_nonzeros:
						max_num_of_nnz = current_num_nnz
						l1_penalty_min = l1_penalty
					else:
						min_num_of_nnz = current_num_nnz
						l1_penalty_max = l1_penalty
						continue_search = False

		lasso_info = {'RSS_best':RSS_best,'L1_best':L1_best,'Best model':best_model,
					'l1_penalty_min':l1_penalty_min,'l1_penalty_max':l1_penalty_max,
					'max_num_of_nnz':max_num_of_nnz,'min_num_of_nnz':min_num_of_nnz}

		return lasso_info

	def compute_ro(self, i, feature_matrix, output, weights):
		""" whenever ro[i] falls between -l1_penalty/2 and l1_penalty/2,
		 the corresponding weight w[i] is sent to zero 
		 ro[i] = SUM[ [feature_i]*(output - prediction + w[i]*[feature_i]) ] """
		prediction = np_utils.predict_output(feature_matrix,weights)
		# Numpy vector for feature_i
		feature_i = feature_matrix[:,i]
		ro_i = (feature_i * (output - prediction + weights[i] * feature_i)).sum()

		return ro_i

	def lasso_coordinate_descent_step(self, i, feature_matrix, output, weights, l1_penalty):
		""" cyclical coordinate descent with normalized features 
		   where we cycle through coordinates 0 to (d-1) in order
		   and assume the features were normalized. 
			       ┌ (ro[i] + lambda/2)     if ro[i] < -lambda/2
			w[i] = ├ 0                      if -lambda/2 <= ro[i] <= lambda/2
			       └ (ro[i] - lambda/2)     if ro[i] > lambda/2 """

		ro_i = self.compute_ro(i, feature_matrix, output, weights)

		if i == 0: # intercept -- do not regularize
			new_weight_i = ro_i
		elif ro_i < -l1_penalty / 2.:
			new_weight_i = ro_i + l1_penalty / 2
		elif ro_i > l1_penalty / 2.:
			new_weight_i = ro_i - l1_penalty / 2
		else:
			new_weight_i = 0.

		return new_weight_i

	def lasso_cyclical_coordinate_descent(self, feature_matrix, output, weights, l1_penalty, tolerance):
		""" cyclical coordinate descent where we optimize 
			coordinates 0, 1, ..., (d-1) in order and repeat. 
			self.lasso_coordinate_descent_step: optimizes the cost function over a single coordinate
		"""
		converged = False
		while not converged:
			max_steps = 0
			for i in range(len(weights)):
				old_weights_i = weights[i]

				weights[i] = self.lasso_coordinate_descent_step(i,feature_matrix,output,weights,l1_penalty)
				#             print "old:%s vs new:%s"%(old_weights_i, weights[i])
				max_steps += abs(old_weights_i - weights[i])
				#             if change_in_coordinate > max_steps:
				#                 max_steps = change_in_coordinate
				if max_steps < tolerance:
					converged = True

		return weights
#==================================================================
#                 Polynomial Regression Functions
#==================================================================
def polynomial_sframe(feature, degree):
	""" create a SFrame to a specific degree"""
	poly_sframe = gp.graphlab.SFrame()
	poly_sframe['power_1']=feature
	if degree > 1:
		for power in range(2, degree+1):
			name = 'power_' + str(power) #column a name
			poly_sframe[name] = feature.apply(lambda x: x**power)

	return poly_sframe

def polynomial_features(data, deg):
	data_copy = data.copy()
	for i in range(1,deg):
		degree = (i + 1)
		data_copy['X%s' % degree] = data_copy['X1'] ** degree
	return data_copy

def polynomial_regression(data, deg, target='Y', l2_penalty=0.,l1_penalty=0., validation_set=None, verbose=False):
	model = gp.graphlab.linear_regression.create(polynomial_features(data,deg),
											  target=target, l2_penalty=l2_penalty,l1_penalty=l1_penalty,
											  validation_set=validation_set,verbose=verbose)
	return model

def polynomial_ridge_regression(data, degree, target='price', l2_penalty=0.,l1_penalty=0., validation_set=None, verbose=False):
	poly_sframe = polynomial_sframe(data['sqft_living'], degree)
	poly_sframe[target] = data[target]
	model = gp.graphlab.linear_regression.create(poly_sframe,
											  target=target, l2_penalty=l2_penalty,l1_penalty=l1_penalty,
											  validation_set=validation_set,verbose=verbose)
	return model, poly_sframe

#==================================================================
#                 K-Fold Cross Validation
#==================================================================
def k_fold_cross_validation(k, dataset, target, features, l2_penalty):
	""" k-fold: starting and ending indices of each segment
		i: current k-fold
		start: start = (n * i) / k
		end: end = (n*(i+1))/k-1
		e.g:  0 (0, 1938)    # i start-end
	          1 (1939, 3878) # i start-end ..... """
	n = len(dataset)
	all_rss = []
	for i in xrange(k):
		start = (n * i) / k
		end = (n * (i + 1)) / k - 1
		validation_k_set = dataset[start:end + 1]
		training_set = dataset[0:start].append(dataset[end + 1:])
		model_k = gp.graphlab.linear_regression.create(training_set, target=target, features=features,
													  l2_penalty=l2_penalty,l1_penalty=0.,
													  validation_set=None,verbose=None)
		rss_k = get_model_residual_sum_of_squares(model_k,validation_k_set,validation_k_set[target])
		#print 'RSS(%s): %s'%(i,rss_k)
		all_rss.append(rss_k)

	return (all_rss)

#==================================================================
#                   Helper Functions
#==================================================================
def compute_RSS(predictions,output):
	"""Residual Sum of Squares (RSS)"""
	residuals = predictions - output
	RSS = (residuals ** 2).sum()
	return (RSS)

def get_model_residual_sum_of_squares(model, data, output):
	"""make predictions and evaluate the model using Residual Sum of Squares (RSS)"""
	predicted_values = model.predict(data)
	return compute_RSS(predicted_values,output)

def get_regression_predictions(input_feature, intercept, slope):
	""" w0:intercept  w1:slope
		yi_predict = (w0 + w1*xi)"""
	return (intercept + slope * input_feature)

def compare_model_predictions(model, dataset, target, target_idx):
	""" target_idx+1 for w0:intercept"""
	predictions = model.predict(dataset)[target_idx+1]
	actual = dataset[target][target_idx]
	return (predictions, actual)

#*******************
#  Plotting helper *
#*******************
def simple_plot(x, y, plot_info,marker='*-', legend_loc='upper left'):
	plt.plot(x,y,marker,label=plot_info['label'])
	plt.legend(loc=legend_loc)
	plt.title(plot_info['title'])
	plt.xlabel(plot_info['x_label'])
	plt.ylabel(plot_info['y_label'])
	if plot_info['yx_axises'] != []:
		plt.axis(plot_info['yx_axises'])

def plot_k_cross_vs_penalty(x, y, marker='k-', penalty='l_2'):
	plt.plot(x,y,marker)
	plt.title('k-fold vs penalty(%s)'%penalty)
	plt.xlabel('$\el%s$ penalty'%penalty)
	plt.ylabel('K-fold cross validation error')
	plt.xscale('log')
	plt.yscale('log')