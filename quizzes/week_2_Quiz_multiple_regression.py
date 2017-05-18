__author__ = 'nadyaK'
__date__ = '04/15/2017'

import ml_graphlab_utils as gp
import ml_numpy_utils as np_utils
import ml_regression_utils as reg
from ml_regression_utils import GrandientDescent
from math import log
import traceback

#==================================================================
#                Multiple Regression Model
#==================================================================
def creat_new_features(dataset):
	""" multiple different features by transformations of existing features 
	   e.g. the log of the square feet or even "interaction" features 
	   such as the product of bedrooms and bathrooms."""

	dataset['bedrooms_squared'] = dataset['bedrooms'].apply(lambda x:x ** 2)
	dataset['bed_bath_rooms'] = dataset['bedrooms']*dataset['bathrooms']
	dataset['log_sqft_living'] = dataset['sqft_living'].apply(lambda x: log(x))
	dataset['lat_plus_long'] = dataset['lat']+dataset['long']

	return dataset

def learning_multiple_models(train_data):
	"""learn the weights for three (nested) models for predicting house prices """

	model_1_features = ['sqft_living','bedrooms','bathrooms','lat','long']
	model_2_features = model_1_features + ['bed_bath_rooms']
	model_3_features = model_2_features + ['bedrooms_squared','log_sqft_living','lat_plus_long']

	multiple_models = {}
	for idx, model_x in enumerate([model_1_features, model_2_features, model_3_features]):
		multiple_models['model_%s'%(idx+1)] = gp.create_linear_regression(train_data,target='price',features=model_x)

	return multiple_models

def multiple_regression_model(train_data, test_data):

	print "\n**********************************"
	print "*    Multiple Regression Model   *"
	print "**********************************\n"

	train_data = creat_new_features(train_data)
	test_data = creat_new_features(test_data)

	print "Quiz_1 (week_2):"

	new_features = ['bedrooms_squared','bed_bath_rooms','log_sqft_living','lat_plus_long']
	for idx,feature in enumerate(new_features):
		print "\nQ%s: %s mean is: %s" % (idx + 1,feature,round(test_data[feature].mean(),2))

	multiple_models = learning_multiple_models(train_data)
	rss_train,rss_test = {},{}
	models_names = ['model_1','model_2','model_3']
	for idx_m,model_i in enumerate(models_names):
		coefficients = multiple_models[model_i].get("coefficients").sort('value',ascending=False)
		# print coefficients #.print_rows(12)
		coeff_dict = gp.convert_sframe_to_simple_dict(coefficients,'name','value')
		print "\nQ%s: coefficient for 'bathrooms' in %s is: %s" % (idx_m + 5,model_i,coeff_dict['bathrooms'])

		rss_train[model_i] = reg.get_model_residual_sum_of_squares(multiple_models[model_i],train_data,train_data['price'])
		rss_test[model_i] = reg.get_model_residual_sum_of_squares(multiple_models[model_i],test_data,test_data['price'])

	print "\nQ8: lowest RSS on TRAINING Data is: %s" % (gp.find_key_min(rss_train))
	print "\nQ9: lowest RSS on TESTING Data is: %s" % (gp.find_key_min(rss_test))

	print '\nRSS-train:%s' % rss_train
	print 'RSS-trest:%s' % rss_test

#==================================================================
#                Grandient Descent Model
#==================================================================
def calculate_weights(gradient, dataset, features, output, parameters):
	initial_weights,step_size,tolerance = parameters
	feature_matrix, output_data = np_utils.get_numpy_data(dataset,features,output)
	weights = gradient.regression_gradient_descent(feature_matrix,output_data,initial_weights,step_size,tolerance)
	return weights

def get_predictions(dataset, features, output, weights):
	feature_matrix, output = np_utils.get_numpy_data(dataset, features, output)
	predictions = np_utils.predict_output(feature_matrix, weights)
	return predictions

def gradient_descent_model(train_data, test_data):
	print "\n**********************************"
	print "*      Gradient Descent Model    *"
	print "**********************************\n"

	gradient = GrandientDescent()

	# Model 1:
	#====================================================
	simple_features, my_output = ['sqft_living'], 'price'
	initial_weights = np_utils.np.array([-47000.,1.])
	step_size, tolerance = 7e-12, 2.5e7
	parameters = [initial_weights,step_size,tolerance]

	# TRAINING data
	simple_weights = calculate_weights(gradient, train_data, simple_features, my_output, parameters)

	# TEST data
	model1_predictions = get_predictions(test_data, simple_features, my_output, simple_weights)
	model1_pred_house1 = round(model1_predictions[0],1)
	print "Quiz_2 (week_2):"
	print "\nQ1: weight for sqft_living (model 1) is: %s" % round(simple_weights[-1],1)
	print "\nQ2: predicted price 1st house in TEST data (model 1) is: %s" % model1_pred_house1

	# Model 2:
	#====================================================
	model_features, my_output_m2 = ['sqft_living', 'sqft_living15'],'price'
	initial_weights_m2 = np_utils.np.array([-100000., 1., 1.])
	step_size_m2,tolerance_m2 = 4e-12, 1e9
	parameters_m2 = [initial_weights_m2,step_size_m2,tolerance_m2]

	# TRAINING data
	estimated_weights = calculate_weights(gradient, train_data, model_features, my_output_m2, parameters_m2)

	# TEST data
	model2_predictions = get_predictions(test_data, model_features, my_output_m2, estimated_weights)
	model2_pred_house1 = round(model2_predictions[0],1)
	print "\nQ3: predicted price 1st house in TEST data (model 2) is: %s" % model2_pred_house1

	true_price_house1 = test_data['price'][0]
	print "\nQ4: True price for the 1st house on the TEST data is: %s" % true_price_house1
	print "\t->diff model1: %s" % abs(model1_pred_house1-true_price_house1)
	print "\t->diff model2: %s" % abs(model2_pred_house1-true_price_house1)

	RSS1 = reg.compute_RSS(model1_predictions,test_data['price'])
	RSS2 = reg.compute_RSS(model2_predictions,test_data['price'])
	print "\nQ5: Which model (1 or 2) has lowest RSS on all of the TEST data"
	print "\t->RSS model1: %s" % RSS1
	print "\t->RSS model2: %s" % RSS2

def main():
	try:
		sales = gp.load_data('../../data_sets/kc_house_data.gl/')
		train_data, test_data = gp.split_data(sales, 0.8)

		multiple_regression_model(train_data,test_data)

		gradient_descent_model(train_data,test_data)

	except Exception as details:
			print "Error >> %s" % details
			traceback.print_exc()

if __name__ == "__main__":
	main()