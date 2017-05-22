__author__ = 'nadyaK'
__date__ = '05/07/2017'

import ml_graphlab_utils as gp
import ml_numpy_utils as np_utils
import matplotlib.pyplot as plt
import traceback

def get_normalized_datasets(train, test, validation, feature_list):
	features_train,output_train = np_utils.get_numpy_data(train,feature_list,'price')
	features_test,output_test = np_utils.get_numpy_data(test,feature_list,'price')
	features_valid,output_valid = np_utils.get_numpy_data(validation,feature_list,'price')

	features_train,norms = np_utils.normalize_features(features_train) # normalize training set features (columns)
	features_test = features_test / norms # normalize test set by training set norms
	features_valid = features_valid / norms # normalize validation set by training set norms

	return (features_train, features_test, features_valid, output_train, output_valid, output_test)

def closest_distance(nth_house, query_house, features_train):
	closest_dist = {'house':0, 'dist':0}

	for house in xrange(nth_house):
		min_dist=0
		current_dist = np_utils.get_euclidean_distance(query_house,features_train[house])
		if (min_dist < current_dist) :
			min_dist = current_dist
			closest_dist.update({'house': house, 'dist': min_dist})
			# print "house:%s  dist=%s" % (house+1, current_dist)
	return closest_dist

def multiple_predictions_k_nearest_neighbors(k, source, matrix_query, output_values):
	nrows,ncols = matrix_query.shape
	prediction_values = []
	for query in xrange(nrows):
		prediction = np_utils.single_prediction_k_nearest_neighbors(k,source,matrix_query[query],output_values)
		prediction_values.append(prediction)
	return prediction_values

def lowest_predicted_house(k, source, matrix_query, output_values):
	multiple_predictions = multiple_predictions_k_nearest_neighbors(k,source,matrix_query,output_values)
	lowest_predict = None
	lowest_house = ''
	for house,prediction in enumerate(multiple_predictions):
		# print "House %s: %s" % (house,prediction)
		if lowest_predict is None or prediction < lowest_predict:
			lowest_predict = prediction
			lowest_house = house

	return lowest_house, lowest_predict

def plot_RSS_vs_validation_set(k_max, features_train, features_valid, output_train, output_valid):
	rss_all = []
	lowest_rss = None
	lowest_k = 0
	for k in xrange(k_max):
		current_prediction = multiple_predictions_k_nearest_neighbors(k,features_train,features_valid, output_train)
		rss = np_utils.compute_RSS(current_prediction,output_valid)
		rss_all.append(rss)
		if (lowest_rss is None or rss < lowest_rss) and not np_utils.np.isnan(rss):
			lowest_rss = rss
			lowest_k = k

	print "\n\nlowest rss:%s and k=%s" % (lowest_rss,lowest_k)

	kvals = range(1,16)
	plt.title('RSS vs validation set')
	plt.xlabel('k-nearest validation set ')
	plt.ylabel('Residual Sum of Squares (RSS)')
	plt.plot(kvals,rss_all,'bo-')
	plt.savefig('../graphs/RSS_vs_validation.png')
	plt.close()

	print rss_all

def main():
	try:
		print "\n**********************************"
		print "*   k-nearest regression Model   *"
		print "**********************************\n"

		sales = gp.load_data('../../data_sets/kc_house_data_small.gl/')

		feature_list = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition',
			'grade','sqft_above','sqft_basement','yr_built','yr_renovated','lat','long','sqft_living15','sqft_lot15']

		train_and_validation,test = sales.random_split(.8,seed=1)
		train,validation = train_and_validation.random_split(.8,seed=1)
		data_sets = get_normalized_datasets(train, test, validation, feature_list)
		features_train,features_test,features_valid,output_train,output_valid,output_test = data_sets
		query_10house = np_utils.get_euclidean_distance(features_test[0],features_train[9])
		print "\nQ1: Euclidean distance query vs 10th house (training): %s" % (round(query_10house,3))

		query_house = features_test[0]
		closest_dist = closest_distance(9,query_house,features_train)
		print "\nQ2: House closest to the query house (training): %s" % (closest_dist)

		close_dist_test = np_utils.get_euclidean_distance_matrix(features_train,features_test[2])
		# print close_dist_test
		print "\nQ3: House (training) closest to query house (test[2]): %s" % (np_utils.np.argmin(close_dist_test))

		print "\nQ4: Predicted value query train=%s vs test: %s" % (train['price'][382], test['price'][382])

		close_4h = np_utils.find_k_nearest_neighbors(4,features_train,features_test[2])
		print "\nQ5: 4 (training) houses closest to query house: %s" % (close_4h)

		predict_avg_houses = np_utils.single_prediction_k_nearest_neighbors(4,features_train,features_test[2], output_train)
		print "\nQ6: Predict the value of the query house (avg k-nearest): %s" % (predict_avg_houses)

		lowest_house, lowest_predict = lowest_predicted_house(10,features_train,features_test[:10],output_train)
		print "\nQ7: Index-house with query set with lowest predicted value: idx(%s):%s" % (lowest_house, lowest_predict)

		# plot_RSS_vs_validation_set(15,features_train,features_valid,output_train,output_valid)

		current_prediction = multiple_predictions_k_nearest_neighbors(8,features_train,features_test,output_train)
		rss = np_utils.compute_RSS(current_prediction,output_test)
		print "\nQ8: k-nearest with optimal k, RSS on the TEST data: %s\n" % (rss)

	except Exception as details:
		print "Error >> %s" % details
		traceback.print_exc()

if __name__ == "__main__":
	main()