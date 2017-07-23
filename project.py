import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import sys
import re
import os.path
import skimage.io as skio

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression


CUR_DIR = os.path.abspath(os.path.dirname(__file__))
IMG_DATA_PATH = os.path.join(CUR_DIR, "katkam-scaled/*.jpg")
WEATHER_DATA_PATH = os.path.join(CUR_DIR, "yvr-weather/*.csv")

p = re.compile(r'katkam-\d\d\d\d\d\d\d\d\d\d\d\d\d\d')
def path_to_time(path):
	match = p.search(path)  
	filename = match.group()
	year  = filename[7:11]
	month = filename[11:13]
	day  = filename[13:15]  
	time  = filename[15:17] + ':' + filename[17:19]
	return year + '-' + month + '-' + day + ' ' + time

def read_image_data(weather_data):

	print("IMAGE DATA_PATH: ", IMG_DATA_PATH)

	# logic borrowed from  
	# stackoverflow https://stackoverflow.com/questions/39195113
	
	# Images are read in the following order:
	# 2016-06-05-06:00:00
	# 2016-06-05-07:00:00
	# ...
	# 2017-06-21-20:00:00
	# 2017-06-21-21:00:00
	filelist = glob.glob(IMG_DATA_PATH)
	df = pd.DataFrame({'filepath': filelist})

	date_time = [] 
	for i, filepath in enumerate(filelist):
		t = path_to_time(filepath)
		date_time.append(t)

	df['Date/Time'] = date_time
	df = df[df['Date/Time'].isin(weather_data['Date/Time'])]
	filelist = df['filepath'].tolist()
	df.to_csv('image_to_weather_file.csv', index=False)

	imgs = skio.imread_collection(filelist)
	img_data = np.array([np.array(img) for img in imgs])
	print(img_data.shape)
	return img_data

def read_weather_data():

	print("WEATHER DATA_PATH: ", WEATHER_DATA_PATH)

	# logic borrowed from  
	# stackoverflow https://stackoverflow.com/questions/39195113
	filelist = glob.glob(WEATHER_DATA_PATH) 
	weather_data = pd.concat(pd.read_csv(file, index_col=None, header=0, skiprows=16) for file in filelist)
	weather_data = weather_data[weather_data["Weather"].notnull()]
	weather_data = weather_data[["Date/Time", "Year", "Month", "Day", "Time", "Temp (°C)", "Weather"]] 
	return weather_data

def clean_data(image_data, weather_data):

	
	# Image  : Select and filter images only if we have weather data for that date 
	return image_data, weather_data

def main():
	
	weathers = read_weather_data()
	imgs = read_image_data(weathers)
	# imgs, weathers = clean_data(imgs, weathers)
	weathers.to_csv("weather.csv", index=False)
	(weathers['Weather'].drop_duplicates()).to_csv("unique_weather_label", index=False)


if __name__ == '__main__':
	main()