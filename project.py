import numpy as np
import pandas as pd
from clean_data import * 


def main():
	weathers = read_weather_data()
	imgs = read_image_data(weathers)
	# imgs, weathers = clean_data(imgs, weathers)
	
  # Do some training......

  # Report Results......

if __name__ == '__main__':
	main()