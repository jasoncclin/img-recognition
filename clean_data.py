import pandas as pd
import numpy as np
import glob
import sys
import re
import os.path
import skimage.io as skio 

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

	# Find the weather data that match the images....
	weather_data = weather_data[weather_data['Date/Time'].isin(df['Date/Time'])]
	weather_counts = weather_data.groupby(['Weather']).aggregate('count').reset_index()
	weather_data.to_csv("weather.csv", index=False)

	weather_counts.to_csv("weather_count.csv", index=False)
	df.to_csv('valid_image.csv', index=False)
	weather_data.to_csv('valid_weather.csv', index=False)

	filelist = df['filepath'].tolist()
	imgs = skio.imread_collection(filelist)
	img_data = np.array([(np.array(img))for img in imgs])
	return img_data, weather_data

single_labels_dict = {
					 'Mostly Cloudy':'Cloudy', 
					 'Mainly Clear':'Clear',
					 'Rain Showers':'Rain',
					 'Heavy Rain':'Rain',
					 'Moderate Rain Showers':'Rain',
					 'Rain,Drizzle,Fog':'Rain_Fog',
					 'Rain,Fog':'Rain_Fog',
					 'Drizzle':'Rain',
					 'Thunderstorms':'Rain',
					 'Moderate Rain,Fog':'Rain_Fog',
					 'Rain Showers,Fog':'Rain_Fog',
					 'Drizzle,Fog':'Rain_Fog',
					 'Moderate Rain Showers,Fog':'Rain_Fog',
					 'Moderate Rain':'Rain',
					 'Rain,Drizzle':'Rain',
					 'Moderate Rain,Drizzle':'Rain',
					 'Heavy Rain,Fog': 'Rain',
					 'Snow Showers':'Wet_Snow',
					 'Rain Showers,Snow Showers':'Wet_Snow',
					 'Rain,Snow':'Wet_Snow',
					 'Snow,Fog':'Snow',
					 'Freezing Fog':'Fog',
					 'Rain,Snow,Fog':'Snow',
					 'Moderate Snow,Fog':'Snow',
					 'Moderate Snow':'Snow',
					 'Snow,Ice Pellets,Fog':'Hail_Fog',
					 'Ice Pellets':'Hail',
					 'Freezing Rain,Fog':'Rain_Fog',
					 'Rain,Ice Pellets':'Rain',
					 'Rain Showers,Snow Showers,Fog':'Rain_Fog',
					 'Rain Showers,Snow Pellets':'Rain'}

multi_label_dict = {
					 'Clear':  ('Clear',),
					 'Cloudy': ('Cloudy',),
					 'Rain':   ('Rain',),	
					 'Fog':    ('Fog',),
					 'Snow':   ('Snow',),
					 'Mostly Cloudy':('Cloudy',), 
					 'Mainly Clear':('Clear',),
					 'Rain Showers':('Rain',),
					 'Heavy Rain':('Rain',),
					 'Moderate Rain':('Rain',),
					 'Moderate Rain Showers':('Rain',),
					 'Drizzle':('Rain',),
					 'Thunderstorms':('Rain',),
					 'Rain,Drizzle':('Rain',),
					 'Moderate Rain,Drizzle':('Rain',),
					 'Rain,Fog':('Rain','Fog'), 
					 'Heavy Rain,Fog': ('Rain', 'Fog'),
					 'Moderate Rain,Fog':('Rain','Fog'),
					 'Rain,Drizzle,Fog':('Rain','Fog'),
					 'Rain Showers,Fog':('Rain','Fog'),
					 'Drizzle,Fog':('Rain', 'Fog'),
					 'Moderate Rain Showers,Fog':('Rain','Fog'),
					 'Rain Snow': ('Rain','Snow'),
					 'Snow Showers':('Rain','Snow'),
					 'Rain Showers,Snow Showers':('Rain','Snow'), 
					 'Freezing Fog':('Fog',),
					 'Snow,Fog':('Snow','Fog',),
					 'Moderate Snow,Fog':('Snow','Fog'),
					 'Moderate Snow':('Snow',),
					 'Snow,Ice Pellets,Fog':('Snow','Fog'),
					 'Ice Pellets':('Hail',),
					 'Freezing Rain,Fog':('Rain','Fog'),
					 'Rain,Ice Pellets':('Rain'),
					 'Rain,Snow': ('Rain','Snow'),
					 "Rain,Snow,Fog": ('Rain','Snow','Fog'),
					 'Rain Showers,Snow Showers,Fog':('Rain','Fog'),
					 'Rain Showers,Snow Pellets':('Rain','Snow')}
					 
def clean_weather_label(label, isMultilabel):
	label_dict = None
	if isMultilabel:
		label_dict = multi_label_dict
	else:
		label_dict = single_labels_dict

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

