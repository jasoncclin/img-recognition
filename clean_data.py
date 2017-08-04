import pandas as pd
import numpy as np
import glob
import sys
import re
import os.path
import skimage.io as skio 
from dictionary import weather_single_label_dict, weather_multi_label_dict, time_label_dict

MULTILABEL_ENABLED = False
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
	hour  = filename[15:17] + ':' + filename[17:19]
	return year, month, day, hour

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

	# Extract the date/time from the file name
	date_time = [] 
	hour = []
	for i, filepath in enumerate(filelist):
		y,m,d,h = path_to_time(filepath)
		date_time.append(y + '-' + m + '-' + d + ' ' + h)
		hour.append(time_label_dict[h])

	# Find the valid images whose date/time is available in the weather data
	df['Date/Time'] = date_time
	df['Hour'] = hour
	df = df[df['Date/Time'].isin(weather_data['Date/Time'])]


	# Find the weather data that match the images....
	weather_data = weather_data[weather_data['Date/Time'].isin(df['Date/Time'])]
	weather_counts = weather_data.groupby('Weather').aggregate('count').reset_index()
	
	df.to_csv('valid_image.csv', index=False)
	weather_data.to_csv('valid_weather.csv', index=False)
	weather_counts.to_csv("weather_count.csv", index=False)
	
	filelist = df['filepath'].tolist()
	imgs = skio.imread_collection(filelist)
	img_data = np.array([(np.array(img)/255)for img in imgs])
	weather_data = weather_data['Weather'].reset_index(drop=True)
	hour_data = df['Hour'].reset_index(drop=True)

	return img_data, hour_data, weather_data
					 
def clean_weather_label(label, isMultilabel):
	label_dict = None
	if isMultilabel:
		label_dict = weather_multi_label_dict
	else:
		label_dict = weather_single_label_dict

	if label in label_dict:
		label = label_dict[label]
	return label

def read_weather_data():
	print("WEATHER DATA_PATH: ", WEATHER_DATA_PATH)
	# logic borrowed from  
	# stackoverflow https://stackoverflow.com/questions/39195113
	filelist = glob.glob(WEATHER_DATA_PATH) 
	weather_data = pd.concat(pd.read_csv(file, index_col=None, header=0, skiprows=16, quoting=2) for file in filelist)
	weather_data = weather_data[weather_data["Weather"].notnull()]
	weather_data = weather_data[["Date/Time", "Year", "Month", "Day", "Time", "Temp (°C)", "Weather"]] 
	(weather_data["Weather"].drop_duplicates()).to_csv("unique_weather_label", index=False)	
	weather_col = weather_data["Weather"].apply(clean_weather_label, isMultilabel = MULTILABEL_ENABLED)
	weather_data['Weather'] = weather_data['Weather'].astype('object')
	weather_data['Weather'] = weather_col
	return weather_data

