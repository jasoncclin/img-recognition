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
	
	# Images are read in the following order:
	# 
	# 2016-06-05-06:00:00
	# 2016-06-05-07:00:00
	# ...
	# ...
	# 2017-06-21-19:00:00
	# 2017-06-21-20:00:00
	# 2017-06-21-21:00:00

	filelist = glob.glob(DATA_PATH)
	#print(filelist)
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
	return df

def clean_data(image_data, weather_data):
	# Weather: remove NaN 
	weather_data = weather_data[weather_data["Weather"].notnull()]
	weather_data = weather_data[["Date/Time", "Year", "Month", "Day", "Time", "Temp (°C)", "Weather"]] 

	# Image  : Select and filter images only if we have weather data for that date 
	return image_data, weather_data

def main():
	imgs = read_image_data()
	weathers = read_weather_data()
	imgs, weathers = clean_data(imgs, weathers)
	weathers.to_csv("weather.csv", index=False)
	(weathers['Weather'].drop_duplicates()).to_csv("unique_weather_label", index=False)


if __name__ == '__main__':
	main()