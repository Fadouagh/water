###
# Created on Feb. 10, 2019
# Author: Fadoua Ghourabi (fadouaghourabi@gmail.com, https://github.com/Fadouagh)
# This code contains functions designed to process the tweet texts:
# Extracting urls, splitting the words of a tweet
# This code uses nltk and word2vec librairies for python.
###


# Python program to generate word vectors using Word2Vec

# importing all necessary modules
import pandas as pd
import re
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings


warnings.filterwarnings(action = 'ignore')

### readURL: extract a list of urls from a text
def readURL(txt):
    pattern = re.compile(r'(https?:\/\/)(\s)?(www\.)?(\s?)(\w+\.)*([\w\-\s]+\/)*([\w-]+)\/?')
    match = pattern.findall(txt)
    url = []
    if match:
        for m in match:
            url.append(''.join(m)) # urls are tuples of groups identified by the regular expression above.
    return url

### splitTw: split a text using pattern given by option reg or default pattern for identifying words and numbers
# text = "2019-03-05 10:05:58.057954,2019-02-27 03:52:41,Tunisie,visions_tn,135121071,1100604607552720896,Perturbation de la distribution de l’eau potable dans la zone inférieure à Utique ville https://t.co/0wZ7AKliWu https://t.co/jbPB7AjhU8"
# splitTw(text) gives ['2019', '03', '05', '10', '05', '58', '057954', '2019', '02', '27', '03', '52', '41', 'Tunisie', 'visions_tn', '135121071', '1100604607552720896', 'Perturbation', 'de', 'la', 'distribution', 'de', 'l', 'eau', 'potable', 'dans', 'la', 'zone', 'inférieure', 'à', 'Utique', 'ville', 'https', 't', 'co', '0wZ7AKliWu', 'https', 't', 'co', 'jbPB7AjhU8']
# Remark: this code gives rise to more accurate result than sentence.replace('.','').split(' '). For instance, tweets may include special characters for space.
def splitTw(txt,reg=r'\W+'):
    return re.split(reg,txt,flags=re.IGNORECASE)

### matchWord: check whether a word is mentioned in a text.
# This function is used to check city names in tweet text.
def matchWord(word,text):
    pattern1 = word+'[\-\w]+' # example Sakiat-sidi-youssef
    reg1 = re.compile(pattern1)
    pattern2 = word+'[\#\w]+' # example Utique#eau_potable
    reg2 = re.compile(pattern2)
    pattern3 = word+'[\@\w]+' # example Utique@rtci
    reg3 = re.compile(pattern3)
    pattern4 = word+'(https?:\/\/)(\s)?(www\.)?(\s?)(\w+\.)*([\w\-\s]+\/)*([\w-]+)\/?' # example Utiquehttps://www.expressfm.com.tn
    reg4 = re.compile(pattern4)
    if (re.search(word,text)) or (re.search(reg1,text)) or (re.search(reg2,text)) or (re.search(reg3,text)) or (re.search(reg4,text)):
        return word
    else:
        return ""

## for testing matchWord
#text = "2019-03-05 10:05:58.057954,2019-02-27 03:52:41,Tunisie,visions_tn,135121071,1100604607552720896,Perturbation de la distribution de l’eau potable dans la zone inférieure à Utique@rtci https://t.co/0wZ7AKliWu https://t.co/jbPB7AjhU8"
#processed_text = splitTw(text)
##print(re.search(r'Utique',text))
#print(matchWord('Utique',text))


#txt = re.split(r'\W+',text,flags=re.IGNORECASE)
#print(splitTw(text))
#for c in cities:
#    #print(c)
#    if (c in txt):
#        print(c)

#   print(''.join(s))
##  Reads ‘mulberry.txt’ file
#tw_path = open("/Users/basho/fadouaproject/SafeWater/files/twData.csv","r")
#tw_data = pd.read_csv(tw_path, header=0)
##
#text= ' '.join(tw_data.TwContent.tolist())
###
#data = []
##
### iterate through each sentence in the file
#for i in sent_tokenize(text):
#    temp = []
###
###  tokenize the sentence into words
#    for j in word_tokenize(i):
#        temp.append(j.lower())
###
#    data.append(temp)
###
##print(data)
###
#### Create CBOW model
#model1 = gensim.models.Word2Vec(data, min_count = 1, size = 10, window = 5)
###
### Print results
#print("Cosine similarity between 'eau' " +
#      "and 'coupure' - CBOW : ",
#      model1.similarity('eau', 'coupure'))
####
#print("Cosine similarity between 'eau' " +
#      "and 'perturbation' - CBOW : ",
#      model1.similarity('eau', 'perturbation'))
####
#print("Cosine similarity between 'eau' " +
#      "and 'contamination' - CBOW : ",
#      model1.similarity('origami', 'geometry'))
####
#print("Cosine similarity between 'eau' " +
#      "and 'production' - CBOW : ",
#      model1.similarity('origami', 'spatial'))
###
###
#### Create Skip Gram model
###model2 = gensim.models.Word2Vec(data, min_count = 1, size = 10, window = 5, sg = 1)
###
#### Print results
###print("Cosine similarity between 'origami' " + "and 'huzita' - Skip Gram : ",
###      model2.similarity('origami', 'huzita'))
###
###print("Cosine similarity between 'origami' " + "and 'fold' - Skip Gram : ",
###      model2.similarity('origami', 'fold'))
###
###print("Cosine similarity between 'origami' " + "and 'geometry' - Skip Gram : ",
###      model2.similarity('origami', 'geometry'))
###
###print("Cosine similarity between 'origami' " +
###      "and 'spatial' - Skip Gram : ",
###      model1.similarity('origami', 'spatial'))
