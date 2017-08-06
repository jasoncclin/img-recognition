#### Required version of Python : 

* Python 3.6.0 
  (Any versoin of Python 3 should work, but Python 3.6 was used in development)

#### Required Software Libraries: 

* numpy
* pandas 
* matplotlib
* scikit-learn/scikit-image
* glob/os/re/sys

#### Commands: 

	$ Python3 project.py

* It takes approximately 2 hours to train all the models

#### Output files: 

* unique_weather_label.csv
* valid_weather.csv 
* valid_weather_count.csv 
* valid_image.csv

* If Cross Validaion is enabled('__CV_ENABLED__' control flag is defined in project.py, default to __False__)
  * validation_scores_weather.csv
  * validation_scores_hour.csv 

* If Cross Validation is not enabled, that goes to final testing
  * testing_scores_weather.csv
  * testing_scores_hour.csv
  * weather-<model_name>.png(9 of them)
  * time-<model_name>.png (9 of them)

