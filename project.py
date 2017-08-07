# Python modules 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# Sklearn modules 
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Our own python helper functions....
from clean_data import read_weather_data, read_image_data, MULTILABEL_ENABLED
from training_models import training_models, flatten_images
from dictionary import weather_scale_dict

# Enable Cross Validation
CV_ENABLED = False 

def plot_prediction(title, xlabel, ylabel, model_name, y_test, y_test_predicted, isWeather=True):
	df = pd.DataFrame({'truth': y_test, 'prediction': y_test_predicted, 'correctness': y_test == y_test_predicted})
	pred = df.groupby(['truth'])['prediction'].aggregate('count').reset_index()
	correct_pred = df.groupby(['truth'])['correctness'].aggregate('sum').reset_index()
	labels = pred['truth'].values.tolist()
	x = range(1,len(labels) + 1)

	# True:  plt.figure(1)
	# False: plt.figure(0)
	plt.figure(isWeather)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	y1 = pred['prediction'].values
	bar1 = plt.bar(x,y1,label=" All predictions", width=0.5)
	plt.xticks(x, labels,fontsize=7)

	# labels = correct_pred['truth'].values.tolist()
	y2 = correct_pred['correctness'].values
	bar2 = plt.bar(x,y2, color=['magenta'], label="Correct predictions", width=0.5)  
	plt.legend(handles=[bar1, bar2])
	if isWeather:
		name = 'weather'
	else:	
		name = 'time'
	plt.savefig(name + '-' + model_name + '.png')

def main():

	
	weathers = read_weather_data()
	images, y_hours, y_weathers = read_image_data(weathers)

	mlb = MultiLabelBinarizer()
	X = flatten_images(images)

	# Training and Testing Set for Weather 
	X_train, X_test, y_train, y_test = train_test_split(X, y_weathers, test_size=0.2, random_state=1)
	
	# Training and Testing Set for Hours 
	X_train_hour, X_test_hour, y_train_hour, y_test_hour = train_test_split(X, y_hours, test_size=0.2, random_state=1)

	if MULTILABEL_ENABLED:
		y_multilabel = mlb.fit_transform(y_weathers)
		print(list(mlb.classes_))
		X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X, y_multilabel, test_size=0.2, random_state=1)
		
	models = training_models() 	

	# Training & Validating
	if CV_ENABLED: 
		cv_weather_scores = pd.DataFrame()
		cv_hour_scores = pd.DataFrame()
		kFold = 5

		# CV: the weather 
		for name, model in models.items():
			if name == 'ovr_svc_lab_model' and MULTILABEL_ENABLED:
				predicted_weather = cross_val_predict(model, X_train_m, y_train_m, cv=kFold)
			else:
				predicted_weather = cross_val_predict(model, X_train, y_train, cv=kFold)

			score = metrics.accuracy_score(y_train, predicted_weather)
			print(score, name, kFold)
			cv_weather_scores = cv_weather_scores.append({'Score': score, 'Model': name, 'K-Fold CV': kFold}, ignore_index=True)

		# CV: the time
		for name, model in models.items():
			predicted_hour = cross_val_predict(model, X_train_hour, y_train_hour, cv=kFold)
			score = metrics.accuracy_score(y_train_hour, predicted_hour)
			print(score, name, kFold)
			cv_hour_scores = cv_hour_scores.append({'Score': score, 'Model': name, 'K-Fold CV': kFold}, ignore_index=True)

		cv_weather_scores.to_csv("cv_scores_weather.csv")
		cv_hour_scores.to_csv("cv_scores_hour.csv")

	else:
		# Final Testing & Report Results... NO MORE TUNING OF THE MODELS 
		test_weather_scores = pd.DataFrame()
		test_hour_scores = pd.DataFrame()

		# Testing: the weather 
		for name, model in models.items():
			if name == 'ovr_svc_lab_model' and MULTILABEL_ENABLED:
				model.fit(X_train_m, y_train_m)
				y_test_predicted = model.predict(X_test_m)
				score = metrics.accuracy_score(y_test_predicted, y_test_m)
			else:
				model.fit(X_train, y_train)
				y_test_predicted = model.predict(X_test)
				score = metrics.accuracy_score(y_test_predicted, y_test)
	 
			print(score, name)
			test_weather_scores = test_weather_scores.append({'Score': score, 'Model': name}, ignore_index=True)
			plot_prediction('Proportion of correct weather prediction',
											'Weather',
											'# of predictions',
											name,
											y_test,
											y_test_predicted)
		test_weather_scores.to_csv("test_scores_weather.csv")

		# Testing the time
		for name, model in models.items():
			model.fit(X_train_hour, y_train_hour)
			y_test_hour_predicted = model.predict(X_test_hour)
			score = metrics.accuracy_score(y_test_hour_predicted, y_test_hour)
			print(score, name)
			test_hour_scores = test_hour_scores.append({'Score': score, 'Model': name}, ignore_index=True)
			plot_prediction('Proportion of correct time prediction',
											'time',
											'# of predictions',
											name,
											y_test_hour,
											y_test_hour_predicted,
											isWeather=False)
		
		test_hour_scores.to_csv("test_scores_hour.csv")


if __name__ == '__main__':
	main()