__author__ = 'nadyaK'
__date__ = '04/22/2017'

import ml_graphlab_utils as gp
import ml_regression_utils as reg
import matplotlib.pyplot as plt
import traceback

def create_regression_model_by_degree(sales, list_of_degrees, sales_col='sqft_living', target='price'):
	poly_n_data= {}
	for degree in list_of_degrees:
		polyn_data = reg.polynomial_sframe(sales[sales_col], degree)
		features_names = polyn_data.column_names()
		# print features_names
		polyn_data[target] = sales[target]
		model_n = gp.create_linear_regression(polyn_data, target=target, features = features_names)
		# model_n.get("coefficients").print_rows(16)
		power_n = 'power_%s'%degree
		poly_n_data[power_n] = {'model':model_n, 'coefficients':model_n.get("coefficients"),'data': polyn_data}

		# plt.plot(polyn_data[power_n],polyn_data['price'],'.',polyn_data[power_n], model_n.predict(polyn_data),'-')
		# plt.show()
	return poly_n_data

def get_polynomial_regression_by_sets(list_of_degrees, list_of_sets):
	polynomial_regressions = {}
	for idx,sets in enumerate(list_of_sets):
		idx_set = 'set_%s' % (idx + 1)
		polynomial_regressions[idx_set] = create_regression_model_by_degree(sets,list_of_degrees)

	return polynomial_regressions

def select_polynomial_degree(train_data, val_data):
	model_by_degree = {}
	rss_all = {}
	list_of_degrees = range(1,16)
	for degree in list_of_degrees:
		data_n_train = reg.polynomial_sframe(train_data['sqft_living'],degree)
		features_names = data_n_train.column_names()
		data_n_train['price'] = train_data['price']
		model_n = gp.create_linear_regression(data_n_train,target='price',features=features_names)
		data_n_val = reg.polynomial_sframe(val_data['sqft_living'],degree)
		data_n_val['price'] = val_data['price']
		rss_n = reg.get_model_residual_sum_of_squares(model_n,data_n_val,data_n_val['price'])
		rss_all[degree] = rss_n
		# print 'RSS(%s): %s' % (degree,rss_n)
		model_by_degree[degree] = model_n

	return gp.find_key_min(rss_all), model_by_degree

def main():
	try:
		print "\n**********************************"
		print "*  Polynomial Regression Model   *"
		print "**********************************\n"

		sales = gp.load_data('../../data_sets/kc_house_data.gl/')
		train,test = sales.random_split(0.5,seed=0)

		set_1,set_2 = train.random_split(0.5,seed=0)
		set_3,set_4 = test.random_split(0.5,seed=0)

		list_of_degrees = [15] #[1,3,5,15]
		list_of_sets = [set_1,set_2,set_3,set_4]
		polynomial_regressions = get_polynomial_regression_by_sets(list_of_degrees, list_of_sets)

		print "\nQ1: power_15 for all four models:"
		pw_degree = 'power_15'
		for idx,sets in enumerate(list_of_sets):
			idx_set = 'set_%s' % (idx + 1)
			poly_n_coeff = polynomial_regressions[idx_set][pw_degree]['coefficients']
			coeff_dict = gp.convert_sframe_to_simple_dict(poly_n_coeff,'name','value')
			print "\t- %s: %s"%(idx_set, coeff_dict[pw_degree])

		print "\nQ2: fitted lines all look the same plots: FALSE"

		training, test_data = sales.random_split(0.9,seed=1)
		train_data, val_data = training.random_split(0.5,seed=1)

		best_degree, model_by_degree  = select_polynomial_degree(train_data,val_data)
		print "\nQ3: the lowest RSS on Validation data is degree:%s" % best_degree

		data_n_test = reg.polynomial_sframe(test_data['sqft_living'],best_degree)
		data_n_test['price'] = test_data['price']
		rss_n = reg.get_model_residual_sum_of_squares(model_by_degree[best_degree],data_n_test,data_n_test['price'])
		print "\nQ4: RSS on TEST with the degree:%s from Validation data is:%s" % (best_degree,rss_n)

	except Exception as details:
		print "Error >> %s" % details
		traceback.print_exc()

if __name__ == "__main__":
	main()