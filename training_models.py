import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage.color import rgb2lab
from skimage.color import lab2rgb

from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier


def rgb2lab_wrapper(images):   
	x = images.shape[0]
	rgb_images = np.reshape(images, (x, 192,256, 3))
	lab = rgb2lab(rgb_images)
	ret = np.reshape(lab, (x,192*256*3))
	return ret

def flatten_images(images):
	# 192 x 256 x 3 
	d0 = images.shape[0]
	d1 = images.shape[1]
	d2 = images.shape[2]
	d3 = images.shape[3]
	images = np.reshape(images, (d0, d1*d2*d3))	
	return images

def training_models():
	bayes_rgb_model = make_pipeline(
		StandardScaler(),
		PCA(3000,whiten=True),
		GaussianNB()
	)
	bayes_lab_model = make_pipeline(
		FunctionTransformer(func=rgb2lab_wrapper),
		StandardScaler(),
		PCA(3000,whiten=True),
		GaussianNB()
	)    
		
	knn_rgb_model = make_pipeline(
		StandardScaler(),
		KNeighborsClassifier(n_neighbors=10)
	)
		
	knn_lab_model = make_pipeline(
		FunctionTransformer(func=rgb2lab_wrapper),
		StandardScaler(),
		PCA(3000,whiten=True),
		KNeighborsClassifier(n_neighbors=10)
	) 
		
	svc_rgb_model = make_pipeline(
		StandardScaler(),
		PCA(3000,whiten=True),
		SVC(kernel='linear', decision_function_shape='ovr',C=0.000005)
	) 

	svc_lab_model = make_pipeline(
		FunctionTransformer(func=rgb2lab_wrapper),
		StandardScaler(),
		PCA(3000,whiten=True),
		SVC(kernel='linear', decision_function_shape='ovr',C=0.000005)
	) 

	ovr_knn_lab_model = make_pipeline(
		FunctionTransformer(func=rgb2lab_wrapper),
		StandardScaler(),
		PCA(3000,whiten=True),
		OneVsRestClassifier(GaussianNB())
	)

	ovr_knn_lab_model = make_pipeline(
		FunctionTransformer(func=rgb2lab_wrapper),
		StandardScaler(),
		PCA(3000,whiten=True),
		OneVsRestClassifier(KNeighborsClassifier(n_neighbors=10))
	)

	ovr_svc_lab_model = make_pipeline(
		FunctionTransformer(func=rgb2lab_wrapper),
		StandardScaler(),
		PCA(3000,whiten=True),
		OneVsRestClassifier(SVC(kernel='linear',C=0.000005))
	)

	models = {
						"bayes_rgb_model": bayes_rgb_model, 
						"bayes_lab_model": bayes_lab_model,
						"knn_rgb_model": knn_rgb_model, 
						"knn_lab_model": knn_lab_model, 
						"svc_rgb_model": svc_rgb_model, 
						"svc_lab_model": svc_lab_model,
						"ovr_knn_lab_model": ovr_knn_lab_model,
						"ovr_svc_lab_model": ovr_svc_lab_model }

	return models

	







