__author__ = 'nadyaK'
__date__ = '05/01/2017'

from math import log,sqrt
import ml_regression_utils as reg
import ml_graphlab_utils as gp
import ml_numpy_utils as np_utils
from ml_regression_utils import LassoRegression
import traceback

#==================================================================
#                 Quiz-1: Lasso select features
#==================================================================
def creat_new_features(dataset):
	""" multiple different features by transformations of existing features  ."""
	dataset['sqft_living_sqrt'] = dataset['sqft_living'].apply(sqrt)
	dataset['sqft_lot_sqrt'] = dataset['sqft_lot'].apply(sqrt)
	dataset['bedrooms_square'] = dataset['bedrooms'] * dataset['bedrooms']
	dataset['floors'] = dataset['floors'].astype(float)
	dataset['floors_square'] = dataset['floors'] * dataset['floors']
	return dataset

def find_features_with_nonzero_weights(sales, all_features, l1_penalty):
	model = gp.graphlab.linear_regression.create(sales, target='price', features=all_features,
	                                              validation_set=None, verbose=False,
	                                              l2_penalty=0., l1_penalty=l1_penalty)

	# print np_utils.print_coefficients(model)
	nnz_weights = gp.get_nonzero_weights_features(model)
	print "\nQ1: Features were assigned nonzero weights: %s" % (nnz_weights)

def find_rss_on_validation_test(training, validation, all_features, l1_penalties, max_nonzeros):
	RSS_best, L1_best, best_model = None, None, None

	for l1_penalty in l1_penalties:
		current_model = gp.graphlab.linear_regression.create(training, target='price', features=all_features,
		                                              validation_set=None, verbose=False,
		                                              l2_penalty=0., l1_penalty=l1_penalty)

		current_num_nnz = current_model.coefficients['value'].nnz()
		# print "\t\tNon-Zeros: %s" % current_num_nnz

		if current_num_nnz == max_nonzeros:
			predictions = current_model.predict(validation)
			RSS = reg.compute_RSS(predictions,validation['price'])
			print "L1 penalty (%.2f)\t\tRSS=%s" % (l1_penalty,RSS)
			if RSS_best is None or RSS < RSS_best:
				RSS_best = RSS
				L1_best = l1_penalty
				best_model = current_model

	lasso_info = {'RSS_best':RSS_best,'L1_best':L1_best,'Best model':best_model}

	return lasso_info

def quiz1_lasso_to_select_features(lasso, sales):
	sales = creat_new_features(sales)
	all_features = ['bedrooms','bedrooms_square','bathrooms','sqft_living','sqft_living_sqrt','sqft_lot',
		'sqft_lot_sqrt','floors','floors_square','waterfront','view','condition','grade','sqft_above','sqft_basement',
		'yr_built','yr_renovated']
	find_features_with_nonzero_weights(sales,all_features,l1_penalty=1e10)

	# learning regression weights with L1_penalties
	(training_and_validation,testing) = sales.random_split(.9,seed=1)
	(training,validation) = training_and_validation.random_split(0.5,seed=1)

	L1_penalities = np_utils.np.logspace(1,7,num=13)
	model_info = {'L1_penalties':L1_penalities,'target':'price','features':all_features}
	lasso_info = lasso.create_lasso_regression(training,validation,model_info)

	print "\nQ2: Best L1 penalty (%.2f) that minimizes the error (RSS) is:%s" % (
	lasso_info["L1_best"],lasso_info["RSS_best"])
	best_model = lasso_info["Best model"]
	# print best_model.coefficients.print_rows(20)
	print "\nQ3: Using best l1_penalty the model '%s' nonzero values\n" % (best_model.coefficients['value'].nnz())

	max_nonzeros = 7
	l1_penalty_values_nn = np_utils.np.logspace(8,10,num=20)
	model_info.update({'L1_penalties':l1_penalty_values_nn})
	lasso_info_nnz = lasso.create_lasso_regression(training,validation,model_info,max_nonzeros)
	l1_penalty_min,l1_penalty_max = lasso_info_nnz["l1_penalty_min"],lasso_info_nnz["l1_penalty_max"]

	print "\nQ4: Value for l1_penalty_min is:%s  & l1_penalty_max is:%s\n" % (l1_penalty_min,l1_penalty_max)

	l1_penalties = np_utils.np.linspace(l1_penalty_min,l1_penalty_max,20)
	lasso_info_val = find_rss_on_validation_test(training,validation,all_features,l1_penalties,max_nonzeros)

	print "\nQ5: L1 penalty with lowest RSS on VALIDATION set is:%s\n" % (lasso_info_val["L1_best"])

	# print np_utils.print_coefficients(lasso_info_val['Best model'])
	nnz_weights = gp.get_nonzero_weights_features(lasso_info_val['Best model'])
	print "\nQ6: l1_penalty found Features with nonzero weights:\n\t%s" % (nnz_weights)

#==================================================================
#                 Quiz-2: Lasso coordinate descent
#==================================================================
def effects_of_l1_penalty_over_weights(lasso, sales):
	simple_features = ['sqft_living','bedrooms']
	my_output = 'price'
	weights = np_utils.np.array([1.,4.,1.])

	simple_feature_matrix,output,norms = np_utils.get_normalized_data(sales,simple_features,my_output)
	# prediction = np_utils.predict_output(simple_feature_matrix,weights)

	Ro = []
	for i in xrange(len(weights)):
		lasso_ro = lasso.compute_ro(i,simple_feature_matrix,output,weights)
		Ro.append(lasso_ro)
		# print 'Ro_%s: %s' % (i,lasso_ro)

	li_penalities = [1.4e8,1.64e8,1.73e8,1.9e8,2.3e8]
	rho_i = Ro[1:] #0: intercept values
	wi = dict(zip(rho_i,[1,2]))
	w1_non_zero, w2_zero, w1_w2_zero = [], [], []
	for rho in rho_i:
		for l1_penalty in li_penalities:
			#whenever ro[i] falls between -l1_penalty/2 and l1_penalty/2 w[i] is sent to zero
			wi_sent_to_zero = (-l1_penalty / 2.) < rho and rho < (l1_penalty / 2.)

			if not wi_sent_to_zero and wi[rho] == 1:
				# print "L1 penalty:%s  wi[rho] == 1" % (l1_penalty)
				w1_non_zero.append(l1_penalty)
			if wi_sent_to_zero and l1_penalty in w1_non_zero and wi[rho] == 2:
				# print "\tL1 penalty:%s  wi[rho] == 1 and w1_non_zero" % (l1_penalty)
				w2_zero.append(l1_penalty)

			if wi_sent_to_zero and wi[rho] == 1:
				w1_w2_zero.append(l1_penalty)
			# if wi_sent_to_zero and l1_penalty in w1_w2_zero and wi[rho] == 2:
				# print "\t\tL1 penalty:%s is w1_w2_zero & wi[rho] == 2" % (l1_penalty)

	w1_nnz_w2_z = list(set(w1_non_zero) & set(w2_zero))
	print "\nQ1: L1_penalties would not set w[1] zero, but would set w[2] to zero:"
	print "\t-> %s"%w1_nnz_w2_z

	print "\nQ2: L1_penalties would set both w[1] and w[2] to zero:"
	print "\t-> %s"%w1_w2_zero

def evaluate_lasso_coordinate(lasso, sales):
	simple_features = ['sqft_living','bedrooms']
	my_output = 'price'
	initial_weights = np_utils.np.zeros(3)
	l1_penalty = 1e7
	tolerance = 1.0

	feature_matrix_norm,output,norms = np_utils.get_normalized_data(sales,simple_features,my_output)

	weights = lasso.lasso_cyclical_coordinate_descent(feature_matrix_norm,output,initial_weights,l1_penalty,tolerance)
	# print weights

	current_predictions = np_utils.predict_output(feature_matrix_norm,weights)
	RSS = reg.compute_RSS(current_predictions,output)
	print "\nQ3: Lasso-coordinate with normalized dataset RSS is: %s" % RSS

	print "\nQ4: Features assigned a zero weight at convergence: %s" % simple_features[-1]
	print "\t->%s" % weights
	# print np_utils.get_nonzero_weights(weights)

def more_features_with_lasso_coordinate(lasso, sales):
	train_data,test_data = sales.random_split(.8,seed=0)

	all_features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade',
		'sqft_above','sqft_basement','yr_built','yr_renovated']

	feature_matrix_norm,train_output,train_norms = np_utils.get_normalized_data(train_data,all_features,'price')

	initial_weights = np_utils.np.zeros(len(all_features) + 1)

	weights_info,nnz_features = {},{}
	penalty_tolerance = [[1e7,1.0],[1e8,1.0],[1e4,5e5]]
	penalty_str = {1e7:'1e7',1e8:'1e8',1e4:'1e4'}
	print "\nFeatures assigned for Q5,Q6,Q7:"

	for penalty,tolerance in penalty_tolerance:
		weights = lasso.lasso_cyclical_coordinate_descent(feature_matrix_norm,train_output,initial_weights,penalty,
			tolerance)
		# print weights
		weights_normalized = weights / train_norms
		weights_info[penalty] = weights_normalized
		dict_weights = dict(zip(['constant'] + all_features,weights_normalized))
		nnz_features[penalty] = filter(lambda x:dict_weights[x] > 0,dict_weights)
		print "\n\tL1_penalty_%s: %s" % (penalty_str[penalty],nnz_features[penalty])

	print "\nQ8: three models RSS on the TEST data:"
	test_feature_matrix,test_output = np_utils.get_numpy_data(test_data,all_features,'price')
	for penalty,tolerance in penalty_tolerance:
		current_predictions = np_utils.predict_output(test_feature_matrix,weights_info[penalty])
		RSS = reg.compute_RSS(current_predictions,test_output)
		print "\n\tL1_penalty_%s: %s" % (penalty_str[penalty],RSS)

def quiz2_lasso_coordinate(lasso, sales):
	print "\n**********************************"
	print "*      Lasso Coordinate Model    *"
	print "**********************************\n"

	sales['floors'] = sales['floors'].astype(int)

	effects_of_l1_penalty_over_weights(lasso,sales)

	evaluate_lasso_coordinate(lasso,sales)

	more_features_with_lasso_coordinate(lasso,sales)

def main():
	try:
		print "\n**********************************"
		print "*      Lasso Regression Model    *"
		print "**********************************\n"

		sales = gp.load_data('../../data_sets/kc_house_data.gl/')
		lasso = LassoRegression()

		quiz1_lasso_to_select_features(lasso, sales)

		quiz2_lasso_coordinate(lasso, sales)
 
	except Exception as details:
		print "Error >> %s" % details
		traceback.print_exc()

if __name__ == "__main__":
	main()