###
# Created on Feb. 10, 2019
# Author: Fadoua Ghourabi (fadouaghourabi@gmail.com, https://github.com/Fadouagh)
# This code uses twitter api to retrieve water-related incidents in french and arabic.
# Related tweets are stored in csv file after (i) removing the redundant tweets, (ii) extracting location from tweet texts.
###
### Major modification history

import tweepy
import csv
import datetime
from data import cities, ids, ids_path
from txtprocessing import readURL, splitTw, matchWord


### appendCSV: append tweets to a data file twData.csv.
# Parameters:
# apidata: raw twitter data
# file: data file to collect the twitter data
# Output:
# nbr: nbr of added tweets
# Features: Timestamp,TwDate,TwLoc,TwUserName,TwUserID,TwID,TwContent,ContentLoc,urls
# Methods: the function checks for redundancy and do some pre-processing to fill in columns ContentLoc and urls
def appendCSV(apidata,file,ids_list=[],cities_list=[]):
    nbr = 0 # number of tweet collected
    unkn = 0 # number of unknown location
    for tweet in apidata:
        # check whether the tweet has not been stored already
        if not(tweet.id in ids_list):
            sentence = tweet.full_text
                #ss = sentence.replace('.','').split(' ')
            ss = splitTw(sentence) # see txtprocessing.py
            location = []
            # identify the location by extracting name of city from the tweet text
            for city in cities_list:
                if (city in ss) or (matchWord(city,sentence)):
                    location.append(city)
                ## to do: add search in entity hashtag
        
            if len(location) == 0:
                unkn = unkn + 1
            
            urls = readURL(tweet.full_text)
            file.writerow([datetime.datetime.now(),tweet.created_at,tweet.user.location,tweet.user.screen_name,
                           tweet.user.id,tweet.id,tweet.full_text,','.join(location),urls])
            nbr = nbr + 1

    return nbr, unkn

# access token of twitter api (add your account key and token)
# add your access token and account keys

consumer_key =
consumer_secret =
access_token =
access_token_secret = 


# Creating the authentication object
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# Setting your access token and secret
auth.set_access_token(access_token, access_token_secret)
# Creating the API object while passing in auth information
api = tweepy.API(auth)

#### Search for keywords in french
query = "eau potable"
#query = "eau"

# Language code (follows ISO 639-1 standards)
language = "fr"
#language = "jp"

# localisation
loc = "33.339922,10.495868,400km" # Center: Medenine, radius: 400km
#loc = "35.6586111111,139.745555556,600km" # Center: Tokyo, radius: 600km

# Calling the user_timeline function with our parameters
results = api.search(q=query, lang=language, geocode=loc, count=1000,tweet_mode='extended')
#results = api.search()
#results = api.search(lang=language, geocode=loc, count=1000, tweet_mode='extended')

# Open/Create a file to append data
file = open(ids_path, 'a')
#file = open('/Users/basho/fadouaproject/SafeWater/files/test.csv', 'a')
#Use csv Writer
writer = csv.writer(file)

# searching for tweets with parameters
nbr, unkn= appendCSV(results,writer,ids,cities)

print("The number of tweet saved is ",nbr)
print("The number of non-identified locations ",unkn)


#### Search in for keywords in arabic
# The search term to find
query = "ماء الشرب"

# Language code (follows ISO 639-1 standards)
language = "ar"

# localisation
#loc = "33.339922,10.495868,500km" #Medenine + 400km

# searching for tweets with parameters
results = api.search(q=query, lang=language, geocode=loc, count=1000, tweet_mode='extended')

nbr, unkn= appendCSV(results,writer,ids,cities)

file.close()

print("The number of tweet saved is ",nbr)
print("The number of non-identified locations ",unkn)
