###
# Created on Mar. 05, 2019
# Author: Fadoua Ghourabi (fadouaghourabi@gmail.com, https://github.com/Fadouagh )
# This code study the collected tweets by performing basic analysis.
###

import pandas as pd
import gensim
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import string


tw_path = open("/Users/basho/fadouaproject/SafeWater/files/twData.csv","r")
tw_data = pd.read_csv(tw_path, header=0)

tweets = tw_data.TwContent.tolist()
    
#text = '. '.join(tw_data.TwContent)
#
data = []
#
## iterate through each sentence in the file
for i in tweets:
    temp = []
#
##  tokenize the sentence into words
    for j in word_tokenize(i):
        temp.append(j.lower())
#
    data.append(temp)
#print(data)


# uncomment if first-time run
#nltk.download('stopwords')

### Below, we perfomr some statistics using nltk.

# --> NLP pipeline
# 1) extract tweet text.
# original_text = tw_data.TwContent.tolist()
original_text = tw_data.TwContent.values
print("Number of tweets is ", len(original_text))

# 2) normalized text is obtained by (i) removing codes for white space, (ii) romoving urls, and (iii) changing to lower case
normalized_text = [re.sub(r'(https?:\/\/)(\s)?(www\.)?(\s?)(\w+\.)*([\w\-\s]+\/)*([\w-]+)\/?', '', i.lower().replace(u'\xa0', u' ')) for i in original_text]

# 3) combine all tweets
all_content = " ".join(normalized_text)
print("The length of the collected tweets is ",len(all_content))

# 4) tokenize the tweets
tokens = all_content.split()
text = nltk.Text(tokens) # not working properly: example "l'eau,c'est"

# --> Searching the text.
# A concordance view shows every occurrence of a given word, together with some context
# examples of appearance of the word "eau"
#text.concordance("eau")
# examples of appearance of the word "coupure"
text.concordance("coupure")
# Common contexts of more than one word
text.common_contexts(["coupure","eau"]) # output: no common contexts were found ???
text.common_contexts(["approvisionnement","eau"])

# ---> Dispertion plot
# The location of certain words in the tweet texts
#text.dispersion_plot(["eau", "potable","coupure","distribution"])

# ---> Basic statistics
# The length of the text
l = len(text)
print("The length of the text in term of tokens: ",l)
# The size of the vocabulary
vocab = sorted(set(text))
s = len(vocab)
print("The size of the vocabulary used in the text: ",s)
# simple frequency code
#freq = {}
#for v in vocab:
#    if not (v in nltk.corpus.stopwords.words('french')) and not (v in string.punctuation):
#        count = 0
#        for e in text:
#            if v == e or e in v:
#                count = count + 1
#        freq[v] = count
#print("Frequency of vocabulary: ", freq)
#freq_eau = freq["eau"] + freq["d'eau"] + freq["eaux"] + freq["l'eau."]
#print("Frequency of word eau: ", freq_eau/l*100)

# frequency distribution for words of interest
text2 = [w for w in text
         if not (w in nltk.corpus.stopwords.words('french')) and not (w in string.punctuation)]
#fdist = text.vocab()
fdist = nltk.FreqDist(text2)
#fdist = frqtext.vocab()
print(fdist["eau"])

    #fdist = [ w for (w,nb) in text.vocab()
# if not (w[0] in nltk.corpus.stopwords.words('french')) and not (w[0] in string.punctuation)]
#print(fdist)
print("Frequency analysis for 'eau': ", fdist["eau"])
print("Frequency analysis for 'coupure': ", fdist["coupure"])
print("Frequency analysis for 'perturbation': ", fdist["perturbation"])
print("Frequency analysis for 'manque': ", fdist["manque"])

# most common words
print("Number of words: ",len(tokens))
print("Number of unique words: ", len(fdist.keys()))
print("The 10 most common words are ",fdist.most_common(10)) # stopwords included!
common = [w for w in fdist.most_common(50)
          if not (w[0] in nltk.corpus.stopwords.words('french')) and not (w[0] in string.punctuation)]
print("The 50 most common words (stopwords and punctuation are excluded) are ", common) # Warning: "les" is a non-stopword in nltk
# cumulative frequency plot for 50 most frequent words
fdist.plot(50, cumulative = True)
# frequency distribution for 50 most frequent words
fdist.plot(50)
#print(fdist.hapaxes())

# Common words that are not stopwords
print("Common words that are not stopwords: ")
non_stopwords = [w for w in list(fdist.keys())[:50] if w not in nltk.corpus.stopwords.words('french')]
print(non_stopwords)

# frequent collocations in the text
# we want to find the word pairs that occur more often than we would expect based on the frequency of the individual word.
text.collocations() # output include <l'eau potable>, <potable dans>, <litres d'eau>

##### commented out on May 09
#
##### Below we study similarity between relevant keywords. We use a vector space model.
##### Create CBOW model
#model1 = gensim.models.Word2Vec(data, min_count = 1, size = 100, window = 3)
####
#### Print results
#print("Cosine similarity between 'eau' " +
#      "and 'potable' - CBOW : ",
#      model1.similarity('eau', 'potable'))
#####
#print("Cosine similarity between 'eau' " +
#      "and 'perturbation' - CBOW : ",
#      model1.similarity('perturbation','eau'))
#####
#print("Cosine similarity between 'eau' " +
#      "and 'coupure' - CBOW : ",
#      model1.similarity('coupure','eau'))
#####
#print("Cosine similarity between 'eau' " +
#      "and 'approvisionnement' - CBOW : ",
#      model1.similarity('approvisionnement','eau'))
#
####
#print(model1.wv.most_similar(positive = ['eau']))
#
#
##Create Skip Gram model: better results!
#model2 = gensim.models.Word2Vec(data, min_count = 1, size = 10, window = 3, sg = 1)
#
#print("Cosine similarity between 'eau' " + "and 'potable' - Skip Gram : ",
#      model2.similarity('eau', 'potable'))
#
#print("Cosine similarity between 'perturbation' " + "and 'eau' - Skip Gram : ",
#      model2.similarity('perturbation', 'eau'))
#
#print("Cosine similarity between 'coupure' " + "and 'eau' - Skip Gram : ",
#      model2.similarity('coupure', 'eau'))
#
#print("Cosine similarity between 'approvisionnement' " +
#      "and 'eau' - Skip Gram : ",
#      model2.similarity('approvisionnement', 'eau'))
#
####
#print(model2.wv.most_similar(positive = ['eau']))












