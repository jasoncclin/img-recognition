#### Required version of Python : 

* Python 3.6.0 
  (Any version of Python 3 should work, but Python 3.6 was used in development)

#### Required Software Libraries: 

* numpy  (1.11.3)
* pandas (0.19.2)
* matplotlib (2.0.0)
* scikit-image (0.12.3)
* scikit-learn (0.18.1)
* glob/os/re/sys
* If it says tkinter is not found, please run `sudo apt-get install python3-tk`


#### Commands: 

	$ Python3 project.py

* It takes approximately 1 hour to train all the models

#### Output files: 

* unique_weather_label.csv
* valid_weather.csv 
* valid_weather_count.csv 
* valid_image.csv

* If Cross Validaion is enabled('__CV_ENABLED__' control flag is defined in project.py, default to __False__)
  * validation_scores_weather.csv
  * validation_scores_hour.csv 

* If Cross Validation is not enabled, then it goes to final testing
  * testing_scores_weather.csv
  * testing_scores_hour.csv
  * weather-<model_name>.png(9 of them)
  * time-<model_name>.png (9 of them)

