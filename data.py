###
# Created on Feb. 10, 2019
# Author: Fadoua Ghourabi (fadouaghourabi@gmail.com, https://github.com/Fadouagh)
# This code returns two lists: name of municipalities in Tunisia and the ids of past tweets.
# modification May 09, 2019 - cities is a numpy array
###


import pandas as pd
import numpy as np
import os


cities_path = os.path.join("/Users/basho/fadouaproject/SafeWater/files", "Tunisie.csv")
data = pd.read_csv(cities_path, index_col=False)


# cities = data.Municipality.tolist() # amazon turk?
cities = data.Municipality.values
#print(cities.shape)



ids_path = os.path.join("/Users/basho/fadouaproject/SafeWater/files", "twData.csv")
ids_data = pd.read_csv(ids_path, header=0)

#data_path = os.path.join("/Users/basho/fadouaproject/SafeWater", "coupurestest.csv")
#data = pd.read_csv(data_path, index_col=False)
#display(data.head())
#print(data.describe())

#ids = ids_data.TwID.tolist()
ids = ids_data.TwID.values
#print(ids.shape)

#print(ids.tolist())
