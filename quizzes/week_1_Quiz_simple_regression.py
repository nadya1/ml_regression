__author__ = 'nadyaK'
__date__ = '04/09/2017'

import ml_regression_utils as reg
import ml_graphlab_utils as gp
from ml_regression_utils import SimpleLinearRegression
import traceback

def main():
	try:
		sales = gp.load_data('../../data_sets/kc_house_data.gl/')
		train_data, test_data = gp.split_data(sales, 0.8)

		simple_reg = SimpleLinearRegression()

		print "\n**********************************"
		print "* Simple Linear Regression Model *"
		print "**********************************\n"
		sqft_intercept, sqft_slope = simple_reg.simple_linear_regression(train_data['sqft_living'],train_data['price'])
		bedroom_intercept, bedroom_slope = simple_reg.simple_linear_regression(train_data['bedrooms'],train_data['price'])
		print "Predicting house prices using:"
		print "\t- Square feet model: Intercept:%s  &  Slope:%s" % (sqft_intercept, sqft_slope)
		print "\t- Bedroom model:     Intercept:%s  &  Slope:%s" % (bedroom_intercept, bedroom_slope)

		print "\nQuiz (week_1):"
		my_house_sqft = 2650
		estimated_price = reg.get_regression_predictions(my_house_sqft,sqft_intercept,sqft_slope)
		print "\nQ1: Predicted price for a house with %s sqft: %s" % (my_house_sqft,estimated_price)

		rss_prices_on_sqft = simple_reg.get_residual_sum_of_squares(train_data['sqft_living'],train_data['price'],sqft_intercept,sqft_slope)
		print "\nQ2: RSS of predicted prices based on sqft is: %s" % rss_prices_on_sqft

		my_house_price = 800000
		estimated_squarefeet = simple_reg.inverse_regression_predictions(my_house_price,sqft_intercept,sqft_slope)
		print "\nQ3: Estimated sqft for a house worth $%d is: %.3f" % (my_house_price,estimated_squarefeet)

		# Compute RSS when using bedrooms on TEST data:
		rss_prices_on_bedroom_test = simple_reg.get_residual_sum_of_squares(test_data['bedrooms'],test_data['price'],bedroom_intercept,bedroom_slope)
		rss_prices_on_sqrt_test = simple_reg.get_residual_sum_of_squares(test_data['sqft_living'],test_data['price'],sqft_intercept,sqft_slope)

		print "\nQ4: Which model (square feet or bedrooms) has lowest RSS on TEST data?"
		print "\t-> RSS (square feet): %s" % (rss_prices_on_sqrt_test)
		print "\t-> RSS (bedroom): %s" % (rss_prices_on_bedroom_test)

	except Exception as details:
		print "Error >> %s" % details
		traceback.print_exc()

if __name__ == "__main__":
	main()