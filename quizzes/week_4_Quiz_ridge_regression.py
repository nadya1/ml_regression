__author__ = 'nadyaK'
__date__ = '04/25/2017'

import ml_regression_utils as reg
import ml_graphlab_utils as gp
import ml_numpy_utils as np_utils
import traceback
import matplotlib.pyplot as plt
from ml_regression_utils import RidgeRegression

#==================================================================
#               Quiz-1: Ridge Regression & l2_penalty
#==================================================================
def create_ridge_regression_and_plot(data_sets, degree, l2_penalty, power_n):
	axis = "power_%s" % power_n
	weights_per_set = {}
	plot_info = {'title': 'Prediction Linear Ridge-Regresion L2:%s' % l2_penalty,
				 'label':'','x_label':axis,'y_label':'price',
				 'yx_axises':[0,14000,0,8e6]}

	plt.figure(figsize=(10,8))

	for idx, set_n in enumerate(data_sets):
		model,poly_n = reg.polynomial_ridge_regression(set_n,degree,target='price', l2_penalty=l2_penalty)
		set_name = 'Set %s' % (idx+1)
		# print set_name
		# w = np_utils.print_coefficients(model)
		weights_per_set[set_name] = gp.get_model_coefficients_dict(model)[axis]
		x_axis = poly_n[axis]
		plot_info['label'] = 'degree %s fit set:%s' % (degree,idx+1)
		reg.simple_plot(x_axis, model.predict(poly_n), plot_info)

	plt.savefig('../graphs/ridge_regression_l2_%s.png'%str(l2_penalty))
	plt.close()

	return weights_per_set

def quiz_1_ridge_regression(sales):
	l2_small_penalty = 1e-5

	model,poly_sframe = reg.polynomial_ridge_regression(sales,degree=15,target='price',l2_penalty=l2_small_penalty)
	# np_utils.print_coefficients(model)
	coeff_powers = gp.get_model_coefficients_dict(model)

	print "\nQ1: Learned coefficient of feature power_1: %s" % (round(abs(coeff_powers['power_1']),2))

	(semi_split1,semi_split2) = sales.random_split(.5,seed=0)
	(set_1,set_2) = semi_split1.random_split(0.5,seed=0)
	(set_3,set_4) = semi_split2.random_split(0.5,seed=0)

	degree = 15
	data_sets = [set_1,set_2,set_3,set_4]
	w_power1 = 1

	# Small Penalty
	l2_small_penalty = 1e-5
	weights_per_set = create_ridge_regression_and_plot(data_sets,degree,l2_small_penalty,w_power1)
	print "\nQ2: Weights por power_1: %s" % weights_per_set
	print "\nQ2: smallest coefficient L2=%s" % (l2_small_penalty)
	print "\t- Range: Between -1000 and -100"
	print "\nQ3: largest coefficient L2=%s" % (l2_small_penalty)
	print "\t- Range: Between 1000 and 10000"

	#Large Penalty
	l2_large_penalty = 1e5
	l2_weights_per_set = create_ridge_regression_and_plot(data_sets,degree,l2_large_penalty,w_power1)
	print "\nQ4: Weights por power_1: %s" % l2_weights_per_set

	min_set = gp.find_key_min(l2_weights_per_set)
	max_set = gp.find_key_max(l2_weights_per_set)
	print "\nQ4: smallest coeff L2=%s  %s: %s" % (l2_large_penalty,min_set,round(l2_weights_per_set[min_set],2))
	print "\nQ5: largest coeff L2=%s  %s: %s" % (l2_large_penalty,max_set,round(l2_weights_per_set[max_set],2))

def compute_k_fold_cross_validation(k, dataset, target, features, l2_penalties):
	print "computting k-fold validation ....."
	all_rss_avg = []
	for l2_penalty in l2_penalties:
		all_rss = reg.k_fold_cross_validation(k,dataset,target,features,l2_penalty)
		rss_avg = np_utils.np.mean(all_rss)
		# print "L2 (%.2f): %s" % (l2_penalty,rss_avg)
		all_rss_avg.append(rss_avg)
	return all_rss_avg

def selecting_l2_via_cross_validation(train_valid):

	train_valid_shuffled = gp.graphlab.toolkits.cross_validation.shuffle(train_valid,random_seed=1)

	k,target,l2_penalties = 10,'price',np_utils.np.logspace(1,7,num=13)
	poly_sframe = reg.polynomial_sframe(train_valid_shuffled['sqft_living'],degree=15)
	features_list = poly_sframe.column_names()
	poly_sframe[target] = train_valid_shuffled[target]

	l2_rss_avg = compute_k_fold_cross_validation(k,poly_sframe,target,features_list,l2_penalties)

	plt.figure(figsize=(10,8))
	reg.plot_k_cross_vs_penalty(l2_penalties, l2_rss_avg)
	plt.savefig('../graphs/k_fold_vd_penalty_l2.png')
	plt.close()

	all_l2_rss_avg = dict(zip(l2_penalties,l2_rss_avg))

	return all_l2_rss_avg

def quiz_1_selecting_l2_penalty(sales):
	print "\n**********************************"
	print "*        k-Fold validation       *"
	print "**********************************\n"

	(train_valid,test) = sales.random_split(.9,seed=1)
	all_l2_rss_avg = selecting_l2_via_cross_validation(train_valid)
	best_l2_penalty = min(all_l2_rss_avg,key=all_l2_rss_avg.get)
	print "\nQ6: Best L2 penalty via 10-fold validation L2 (%.2f): %s" % (best_l2_penalty,all_l2_rss_avg[best_l2_penalty])

	degree = 15
	model_train_valid,poly_sframe_train_valid = reg.polynomial_ridge_regression(train_valid,degree,target='price',l2_penalty=float(best_l2_penalty))
	poly_sframe_test = reg.polynomial_sframe(test['sqft_living'],degree)
	poly_sframe_test['price'] = test['price']
	rss_n = reg.get_model_residual_sum_of_squares(model_train_valid,poly_sframe_test,poly_sframe_test['price'])
	print "\nQ7: Predictions for degree=%s TEST error (RSS)=%s" % (degree,rss_n)
	print "\t- Between 8e13 and 4e14"

def compute_ridge_regression(ridge, feature_matrix, output, l2_penalties, init_weights):
	step_size = 1e-12
	max_iterations = 1000
	fist_house = 1
	y_axis = []
	ridge_info = {}
	for l2_penalty in l2_penalties:
		iteration,weights = ridge.ridge_regression_gradient_descent(feature_matrix,output, init_weights,
			step_size,l2_penalty,max_iterations,debug=False)
		# print 'L2(%s)\tIteration:%s & Learned weights:%s' % (l2_penalty,iteration,weights)
		y_axis.append(np_utils.predict_output(feature_matrix,l2_penalty))
		ridge_info[l2_penalty]=weights
		# print weights
		print "\n\tcoefficients with regularization (L2:%s) is: %s" % (l2_penalty,round(weights[fist_house],1))

	# x_axis = feature_matrix
	# plt.plot(x_axis,y_axis[0],x_axis,y_axis[1])
	# plt.savefig('../graphs/ridge_reg_learned_weights.png')
	# plt.close()

	return ridge_info

def compute_ridge_rss(weights_list, feature_matrix, test_data):
	for weights_vals in weights_list:
		current_predictions = np_utils.predict_output(feature_matrix,weights_vals)
		RSS1 = reg.compute_RSS(current_predictions,test_data['price'])
		# print 'RSS1: %s' % (RSS1)
		print "\n\tTEST error (RSS) is: %s" % (RSS1)

#==================================================================
#                 Quiz-2: Ridge Gradient Descent
#==================================================================
def quiz_2_ridge_grandient_descent(sales):
	print "\n**********************************"
	print "*     Ridge Gradient Descent     *"
	print "**********************************\n"

	simple_features = ['sqft_living']
	my_output = 'price'
	train_data,test_data = sales.random_split(.8,seed=0)
	(simple_feature_matrix,output) = np_utils.get_numpy_data(train_data,simple_features,my_output)
	(simple_test_feature_matrix,test_output) = np_utils.get_numpy_data(test_data,simple_features,my_output)

	ridge = RidgeRegression()
	l2_no_reg,l2_high_reg = 0,1e11
	initial_weights = np_utils.np.array([0.,0.])
	print "\nQ1 & Q2 coefficients with features: %s" % (simple_features)
	ridge_weights = compute_ridge_regression(ridge,simple_feature_matrix,output,[l2_no_reg,l2_high_reg],initial_weights)
	# print ridge_weights

	print "\nQ3: Line fit with no regularization (l2_penalty=0) is steeper"
	print "\nQ4: high regularization (l2_penalty=1e11)"
	compute_ridge_rss([ridge_weights[l2_high_reg]],simple_test_feature_matrix,test_data)
	print "\t- Between 5e14 and 8e14"

	more_features = ['sqft_living','sqft_living15']
	initial_w_morefeatures = np_utils.np.array([0.0,0.0,0.0])
	(more_feature_matrix,output_more_features) = np_utils.get_numpy_data(train_data,more_features,my_output)
	(more_test_feature_matrix,test_output_more) = np_utils.get_numpy_data(test_data,more_features,my_output)

	print "\nQ5 & Q6 coefficients with features: %s" % (more_features)
	ridge_morefeatures = compute_ridge_regression(ridge,more_feature_matrix,output_more_features,
		[l2_no_reg,l2_high_reg],initial_w_morefeatures)

	print "\nQ7: using all zero weights with features: %s" % (simple_features)
	compute_ridge_rss([initial_w_morefeatures],more_test_feature_matrix,test_data)
	print "\t-Between 1e15 and 3e15"

	num_of_house = 1#5
	print "\nQ8: Which model makes better predictions for 1st house:"
	for l2_penalty in [l2_no_reg,l2_high_reg]:
		print "L2:%s:" % l2_penalty
		current_predictions = np_utils.predict_output(more_test_feature_matrix,ridge_morefeatures[l2_penalty])
		for house_predict in range(num_of_house):
			pred,real = current_predictions[house_predict],test_data['price'][house_predict]
			print '\t\t(predict) %s vs %s (real)  diff: %s' % (pred,real,real - pred)

def main():
	try:
		print "\n**********************************"
		print "*     Ridge Regression Model     *"
		print "**********************************\n"

		sales = gp.load_data('../../data_sets/kc_house_data.gl/')
		sales_q1 = sales.sort(['sqft_living','price'])

		quiz_1_ridge_regression(sales_q1)

		quiz_1_selecting_l2_penalty(sales_q1)

		quiz_2_ridge_grandient_descent(sales)

	except Exception as details:
		print "Error >> %s" % details
		traceback.print_exc()

if __name__ == "__main__":
	main()