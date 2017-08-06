weather_scale_dict = {
					 'Clear': 2,
					 'Cloudy': 4,
					 'Fog':  6,
					 'Rain': 8,
					 'Snow': 10}


weather_single_label_dict = {
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
					 'Drizzle,Fog':'Fog',
					 'Moderate Rain Showers,Fog':'Rain_Fog',
					 'Moderate Rain':'Rain',
					 'Rain,Drizzle':'Rain',
					 'Moderate Rain,Drizzle':'Rain',
					 'Heavy Rain,Fog': 'Rain_Fog',
					 'Snow Showers':'Snow',
					 'Rain Showers,Snow Showers':'Rain_Snow',
					 'Rain,Snow':'Rain_Snow',
					 'Snow,Fog':'Snow_Fog',
					 'Freezing Fog':'Fog',
					 'Rain,Snow,Fog':'Rain_Snow_Fog',
					 'Moderate Snow,Fog':'Snow_Fog',
					 'Moderate Snow':'Snow',
					 'Freezing Rain,Fog':'Rain_Fog',
					 'Rain,Ice Pellets':'Rain',
					 'Rain Showers,Snow Showers,Fog':'Rain_Snow_Fog',
					 'Rain Showers,Snow Pellets':'Rain_Snow'}

weather_multi_label_dict = {
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

time_label_dict = {
					 '06:00':'morning',
					 '07:00':'morning',
					 '08:00':'morning',
					 '09:00':'morning',
					 '10:00':'morning',
					 '11:00':'noon',
					 '12:00':'noon',
					 '13:00':'noon',
					 '14:00':'noon',
					 '15:00':'afternoon',
					 '16:00':'afternoon',
					 '17:00':'afternoon',
					 '18:00':'afternoon',
					 '19:00':'evening',
					 '20:00':'evening',
					 '21:00':'evening'}