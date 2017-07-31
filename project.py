# Python modules 
import numpy as np
import pandas as pd

# Sklearn modules 
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Our own python helper functions....
from clean_data import read_weather_data, read_image_data, MULTILABEL_ENABLED
from training_models import training_models, flatten_images


def main():
	weathers = read_weather_data()
	images, weathers = read_image_data(weathers)
	
	mlb = MultiLabelBinarizer()
	X = flatten_images(images)
	y = weathers['Weather'].reset_index(drop=True)  
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

	if MULTILABEL_ENABLED:
		y_multilabel = mlb.fit_transform(y)
		print(list(mlb.classes_))
		X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X, y_multilabel, test_size=0.2, random_state=1)

	# Training & Validating
	models = training_models() 
	scores = pd.DataFrame()
	kFold = 5

	for name, model in models.items():
		if name == 'ovr_svc_lab_model' and MULTILABEL_ENABLED:
			predicted = cross_val_predict(model, X_train_m, y_train_m, cv=kFold)
		else:
			predicted = cross_val_predict(model, X_train, y_train, cv=kFold)

		score = metrics.accuracy_score(y_train, predicted)
		print(score, name, kFold)
		scores = scores.append({'Score': score, 'Model': name, 'K-Fold CV': kFold}, ignore_index=True)
	
	scores.to_csv("validation_scores.csv")
	# Final Testing & Report Results... NO MORE TUNING OF THE MODELS 
	

if __name__ == '__main__':
	main()