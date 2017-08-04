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

def single_label_to_scale(label):
	scale = 0
	if label in weather_scale_dict:
 		scale = weather_scale_dict[label]
	return scale 

def main():
	weathers = read_weather_data()
	images, y_hours, y_weathers = read_image_data(weathers)

	mlb = MultiLabelBinarizer()
	X = flatten_images(images)
	
	# Training and Testing Set for Weather 
	X_train, X_test, y_train, y_test = train_test_split(X, y_weathers, test_size=0.2, random_state=1)
  
	# X_length = X_train.shape[0]
	# y_scale  = y_train.apply(single_label_to_scale)
	# plt.plot(range(X_length), y_scale)
	# plt.show()

	# Training and Testing Set for Hours 
	X_train_hour, X_test_hour, y_train_hour, y_test_hour = train_test_split(X, y_hours, test_size=0.2, random_state=1)

	if MULTILABEL_ENABLED:
		y_multilabel = mlb.fit_transform(y_weathers)
		print(list(mlb.classes_))
		X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X, y_multilabel, test_size=0.2, random_state=1)
  




	# Training & Validating
	models = training_models() 
	weather_scores = pd.DataFrame()
	hour_scores = pd.DataFrame()
	kFold = 5

	# Predicting the weather 
	# for name, model in models.items():
	# 	if name == 'ovr_svc_lab_model' and MULTILABEL_ENABLED:
	# 		predicted_weather = cross_val_predict(model, X_train_m, y_train_m, cv=kFold)
	# 	else:
	# 		predicted_weather = cross_val_predict(model, X_train, y_train, cv=kFold)

	# 	score = metrics.accuracy_score(y_train, predicted_weather)
	# 	print(score, name, kFold)
	# 	weather_scores = weather_scores.append({'Score': score, 'Model': name, 'K-Fold CV': kFold}, ignore_index=True)

	name = "ovr_svc_lab_model"
	# predicted_weather = cross_val_predict(models[name], X_train, y_train, cv=kFold)
	# score = metrics.accuracy_score(y_train, predicted_weather)	
	# print(score, name, kFold)

	# models[name].fit(X_train, y_train)
	# score = metrics.accuracy_score(models[name].predict(X_test), y_test)
	# print(score)


	models[name].fit(X_train_hour, y_train_hour)
	score = metrics.accuracy_score(models[name].predict(X_test_hour), y_test_hour)
	print(score)


	# Predicting the time
	# for name, model in models.items():
	# 	predicted_hour = cross_val_predict(model, X_train_hour, y_train_hour, cv=kFold)
	# 	score = metrics.accuracy_score(y_train_hour, predicted_hour)
	# 	print(score, name, kFold)
	# 	hour_scores = hour_scores.append({'Score': score, 'Model': name, 'K-Fold CV': kFold}, ignore_index=True)


	# predicted_hour = cross_val_predict(model, X_train_hour, y_train_hour, cv=kFold)
	# score = metrics.accuracy_score(y_train_hour, predicted_hour)
	# print(score, name, kFold)
	# hour_scores = hour_scores.append({'Score': score, 'Model': name, 'K-Fold CV': kFold}, ignore_index=True)

	# weather_scores.to_csv("validation_scores_weather.csv")
	# hour_scores.to_csv("validation_scores_hour.csv")


	# Final Testing & Report Results... NO MORE TUNING OF THE MODELS 
	

if __name__ == '__main__':
	main()