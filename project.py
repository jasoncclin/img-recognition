# Python modules 
import numpy as np
import pandas as pd

# Sklearn modules 
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Our own python helper functions....
from clean_data import read_weather_data, read_image_data
from training_models import training_models, flatten_images


def main():
	weathers = read_weather_data()
	images, weathers = read_image_data(weathers)

	X = flatten_images(images)
	y = weathers['Weather'].reset_index(drop=True)
	print(X.shape)
	print(y.shape)
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

	# Training & Validating
	models = training_models()
	scores = [] 
	kFold = 10

	# predicted = cross_val_predict(models[5], X_train, y_train, cv=kFold)
	# score = metrics.accuracy_score(y_train, predicted)
	
	score = cross_val_score(models[3], X_train, y_train, cv=kFold)
	scores.append(score)
	print(score)

	# for i, m in enumerate(models):
	# 	accurary_score = cross_val_score(m, X_train, y_train, cv=k)
	#   scores.append(accurary_score)

 #  # Report validation scores and further do some further tuning to the models 
 #  print(OUTPUT_TEMPLATE.format(
 #  	k=kFold,
 #  	bayes_rgb=score[0],
 #  	bayes_lab=score[1],
 #  	knn_rgb=score[2],
 #  	knn_lab=score[3],
 #  	svm_rgb=score[4],
 #  	svm_lab=score[5]
 #  ))    

	# Final Testing & Report Results... NO MORE TUNING OF THE MODELS 


if __name__ == '__main__':
	main()