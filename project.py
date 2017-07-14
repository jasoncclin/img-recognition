import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import sys
import os.path
import skimage.io as skio

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

CUR_DIR = os.path.abspath(os.path.dirname(__file__))

def read_image_data():

	DATA_PATH = os.path.join(CUR_DIR, "katkam-scaled/*.jpg")
	print("IMAGE DATA_PATH: ", DATA_PATH)

	# logic borrowed from  
	# stackoverflow https://stackoverflow.com/questions/39195113
	filelist = glob.glob(DATA_PATH)
	imgs = skio.imread_collection(filelist)
	x = np.array([np.array(img) for img in imgs])
	print(x.shape)
	return x 

def read_weather_data():

	DATA_PATH = os.path.join(CUR_DIR, "yvr-weather/*.csv")
	print("WEATHER DATA_PATH: ", DATA_PATH)
	
	# logic borrowed from  
	# stackoverflow https://stackoverflow.com/questions/39195113
	filelist = glob.glob(DATA_PATH) 
	df = pd.concat(pd.read_csv(file, index_col=None, header=0, skiprows=16) for file in filelist)
	print (df[['Date/Time', 'Weather']])
	return df

def clean_data(image_data, weather_data):
	# Weather: remove NaN 
	# Image  : Select and filter images only if we have weather data for that date 
	return None

def main():
	imgs = read_image_data()
	weathers = read_weather_data()
  clean_data(imgs, weathers)

if __name__ == '__main__':
	main()