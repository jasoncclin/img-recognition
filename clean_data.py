import pandas as pd
import numpy as np
import glob
import sys
import re
import os.path
import skimage.io as skio

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

  # Extract the date/time from the file name
	date_time = [] 
	for i, filepath in enumerate(filelist):
		t = path_to_time(filepath)
		date_time.append(t)

  # Find the valid images whose date/time is available in the weather data
	df['Date/Time'] = date_time
	df = df[df['Date/Time'].isin(weather_data['Date/Time'])]
	filelist = df['filepath'].tolist()
	
	# Output the valid images to a csv file 
	df.to_csv('valid_image.csv', index=False)

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
	weather_data['Weather'] = weather_data['Weather'].apply(clean_weather_label) 
	weather_data.to_csv("weather.csv", index=False)
	(weather_data['Weather'].drop_duplicates()).to_csv("unique_weather_label", index=False)
	return weather_data



clean_labels_dict = {'Mostly Cloudy':'Cloudy', 
					 'Mainly Clear':'Clear',
					 'Rain Showers':'Rain',
					 'Heavy Rain':'Rain',
					 'Moderate Rain Showers':'Rain',
					 'Rain,Drizzle,Fog':'Rain and Fog',
					 'Rain,Fog':'Rain and Fog',
					 'Drizzle':'Rain',
					 'Thunderstorms,Rain Showers':'Thunderstorms',
					 'Moderate Rain,Fog':'Rain and Fog',
					 'Rain Showers,Fog':'Rain and Fog',
					 'Drizzle,Fog':'Rain and Fog',
					 'Moderate Rain Shower,Fog':'Rain and Fog',
					 'Moderate Rain':'Rain',
					 'Rain,Drizzle':'Rain',
					 'Moderate Rain,Drizzle':'Rain',
					 'Heavy Rain,Fog': 'Rain',
					 'Snow Showers':'Wet Snow',
					 'Rain Showers,Snow Showers':'Wet Snow',
					 'Rain,Snow':'Wet Snow',
					 'Snow,Fog':'Snow and Fog',
					 'Freezing Fog':'Fog',
					 'Rain,Snow,Fog':'Wet Snow and Fog',
					 'Moderate Snow,Fog':'Wet Snow and Fog',
					 'Moderate Snow':'Snow',
					 'Snow,Ice Pellets,Fog':'Hail and Fog',
					 'Ice Pellets':'Hail',
					 'Freezing Rain,Fog':'Rain and Fog',
					 'Rain,Ice Pellets':'Rain and Hail',
					 'Rain Showers,Snow Showers,Fog':'Wet Snow and Fog',
					 'Rain Showers,Snow Pellets':'Rain and Hail'}



def clean_weather_label(label):
	if label in clean_labels_dict:
		label = clean_labels_dict[label]
		
	return label


