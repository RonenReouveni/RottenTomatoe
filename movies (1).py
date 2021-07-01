import pandas as pd 
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import re
from sklearn.naive_bayes import ComplementNB        #import model
from sklearn.model_selection import cross_val_score, cross_val_predict #import cross validation 
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer #import module 
from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM , Dense, Dropout
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.layers import Embedding
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


movies = pd.read_csv('/Users/ronenreouveni/Desktop/NLP_movie/train.tsv', sep='\t') #import data
sentList = movies['Sentiment'] #extract labels
train = movies.filter(['Phrase','Sentiment'], axis=1) #create a new df from relevant columns
train['Phrase'] = train['Phrase'].str.lower() #lowercase all the text
PhraseList = list(train['Phrase']) #convert type series into list

print(train.head)

#use seaborn to visualize our distribution of sentimet
distro = train.Sentiment.value_counts()
sns.barplot(distro.index, distro.values)



#https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
#function to expand all contractions
#this does not catch everything, but its pretty close
def decontracted(phrase):
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

#use map function to apply functin to each element 
PhraseList = list(map(decontracted, PhraseList))


#extract subjectivity scores by phrase for later use 
#.sentiment[0] is polarity while .sentiment[1] is subjectivity
subjectivityFeature = [] #create empty list
for phrase in PhraseList:
    subjectivityFeature.append(TextBlob(phrase).sentiment[1]) #loop and append score for each phrase



#tokenize each phrase 
tokenized_sents = [word_tokenize(i) for i in PhraseList]

#this next section contains advanced feature extraction
#In depth explanation can be found in the report 
#a demo with examples is found in the helper.py 
#helper.py is not connected to this script but only used to highlight behavior of the custom sentiment calculator

#define list of 'flippers'
flippers = ['but', 'however', 'although']

#define list of negators 
negationwords = ['no', 'not', 'never', 'none', 'nowhere', 'nothing', 'noone', 'rather', 'hardly', 'scarcely', 'rarely', 'seldom', 'neither', 'nor']


#this function calculates sentiment for each word
#it uses an idea of a 'state', that is updated with each additional token or word 
#this allows me to edit the state based on specific criteria 
#this code is completely custom and created by me, Ronen Reouveni 

def customSentiment(tokenizedSentences): #define function
    phraseS = [] #create empty list
    for sentence in tokenizedSentences: #begin outer loop
        state = 0 #reset state to zero
        indexes = iter(range(len(sentence))) #create iterable so I can utilize indexes and next 
        for i in indexes: #begin inner loop 
            if sentence[i] in flippers: #if the word is in the flippers list reset state to 0
                state = 0
            state += (TextBlob(sentence[i]).sentiment[0]) #if not, sum the polarity
            if sentence[i] in negationwords: #if the word is a negation word flip the sign of the next word
                try: #need try incase the phrase is one word
                    state -= (TextBlob(sentence[i+1]).sentiment[0]) #if the previous is a negation word subtract from state
                    next(indexes) #skip the next word because we looked ahead with i+1
                except Exception: #if we get an error just pass 
                    pass
        phraseS.append(state) #append the final state value and move onto the next phrase
    return phraseS

phraseSentiment = customSentiment(tokenized_sents) #call the function on our full tokenized list
print(len(phraseSentiment)) #make sure we have 156,060 values returned by the function 


#the sentiment is both positive and negative numbers 
#ComplementNB only accepts positive numbers 
#We therefore must normalize the list and make the smallest numbers 0
def normalized(myList):
    normalSents = []
    for x in myList:
        normalSents.append((x-min(myList))/(max(myList)-min(myList)))
    return normalSents


phraseSentiment = normalized(phraseSentiment)



#create a function that extracts features 
#find detailed explanation in report 
#function takes text, minimum n_gram amount, max n_gram, min docs, max docs, and boolean of stop
# text = text to be used 
# if mingram is set to 1, only unigrams, 
#if maxgram set to 3 and mingram set to 1, we get unigrams, bigrams, and trigrams
#minDocs is minimum number of documents a n_gram needs to be found in to be addmited as a feature 
#maxDocs is maximum number of is maximum number of documents an n_gram can be in to be addmited as a feature
#stop is a boolean, if true, we remove stopwords, if false, we leave in place
def createFeatureSet(text, minGram, maxGram, minDocs, maxDocs, stop):
    if stop == True: #if we want stop words removed it fires this set 
        #define vectorizer with the desired params
        vec = CountVectorizer(ngram_range=(minGram,maxGram), min_df = minDocs ,max_df = maxDocs, stop_words='english') 
        X = vec.fit_transform(text) #fit it on text
        return(pd.DataFrame(X.toarray(), columns=vec.get_feature_names())) #return it to a df
    else: #if we want to ignore stopwords it runs this set
        vec = CountVectorizer(ngram_range=(minGram,maxGram), min_df = minDocs ,max_df = maxDocs) 
        X = vec.fit_transform(text) 
        return(pd.DataFrame(X.toarray(), columns=vec.get_feature_names()))


#define target names, will be used later
target_names = ['neg', 'some_neg', 'neutral' , 'som_pos', 'pos']

#Model 1 
#unigrams, no stop word removal 
#1128 words chosen 
df_uni = createFeatureSet(PhraseList, 1,1,100,750,False) #create bag of words

print(df_uni.shape) #view shape of bag of words frame
print(df_uni.head) #view head of bag of words frame

clf = ComplementNB() #instantiate model 
scores = cross_val_score(clf, df_uni, sentList, cv=5) #call model with 5 folds 
print('Model 1')
print(scores) #view all 5 scores
print((sum(scores))/5) #calculate mean score
y_pred = cross_val_predict(clf, df_uni, sentList, cv=5) #use cross validation to predict 
conf_mat = confusion_matrix(sentList, y_pred) #construct confusion matrix
print('Model 1')
print(conf_mat) #print confusion matrix
print(classification_report(sentList, y_pred, target_names=target_names)) #show all stats for each class
clf.fit(df_uni, sentList) #fit entire model
#plot a nice looking confusion matrix
print('Model 1')
(plot_confusion_matrix(clf, df_uni, sentList, normalize= 'true')) #note, normalize = true, means we get a nice percentage, not raw count




#Model 2
#unigrams, bigrams, and trigrams, 
#no stopword removal 
#1669 predictors
df_uni_bi_tri = createFeatureSet(PhraseList, 1,3,100,750,False)
print(df_uni_bi_tri.shape)
clf = ComplementNB() #instantiate model 
scores = cross_val_score(clf, df_uni_bi_tri, sentList, cv=5) #call model with 5 folds 
print('Model 2')

print(scores)
print((sum(scores))/5)
y_pred = cross_val_predict(clf, df_uni_bi_tri, sentList, cv=5)
conf_mat = confusion_matrix(sentList, y_pred)
print('Model 2')

print(conf_mat)
print(classification_report(sentList, y_pred, target_names=target_names))
clf.fit(df_uni_bi_tri, sentList)
print('Model 2')

(plot_confusion_matrix(clf, df_uni_bi_tri, sentList, normalize= 'true'))


 


#Model 3
#unigrams, bigrams trigrams, and custom sentiment, and subjectivity
#the custom sentiment uses negation and other advanced programming, see report 
#no stopwords removed
#2392 predictors 
df_full = createFeatureSet(PhraseList, 1,3,75,800,False) 
print(df_full.shape)
df_full['custom'] = phraseSentiment
df_full['subjectivity'] = subjectivityFeature


clf = ComplementNB() #instantiate model 
scores = cross_val_score(clf, df_full, sentList, cv=5) #call model with 5 folds 
print('Model 3')
print(scores)
print((sum(scores))/5)
y_pred = cross_val_predict(clf, df_full, sentList, cv=5)
conf_mat = confusion_matrix(sentList, y_pred)
print('Model 3')
print(conf_mat)
print(classification_report(sentList, y_pred, target_names=target_names))
clf.fit(df_full, sentList)
(plot_confusion_matrix(clf, df_full, sentList, normalize= 'true'))




#Model 3.5
#same as model above but stopwords are removed (does worse than the above model)
df_full_stop = createFeatureSet(PhraseList, 1,3,75,800,True) 
print(df_full_stop.shape)
df_full_stop['custom'] = phraseSentiment
df_full_stop['subjectivity'] = subjectivityFeature

clf = ComplementNB() #instantiate model 
scores = cross_val_score(clf, df_full_stop, sentList, cv=5) #call model with 5 folds
print('Model 3.5')

print(scores)
print((sum(scores))/5)
y_pred = cross_val_predict(clf, df_full_stop, sentList, cv=5)
conf_mat = confusion_matrix(sentList, y_pred)
print('Model 3.5')
print(conf_mat)
print(classification_report(sentList, y_pred, target_names=target_names))
clf.fit(df_full_stop, sentList)
(plot_confusion_matrix(clf, df_full_stop, sentList, normalize= 'true'))









#Model 4
#BEWARE very long runtime, 3+ hours so its commented out 
#uses same data as Model 3, the best Naive Bayes classifier

#clf_rf = RandomForestClassifier()
#scores = cross_val_score(clf_rf, df_full, sentList, cv=3) #call model with 3 folds 
#print('Model 4')
#print(scores)
#print((sum(scores))/3)
#y_pred = cross_val_predict(clf_rf, df_full, sentList, cv=5)
#conf_mat = confusion_matrix(sentList, y_pred)
#print('Model 4')
#print(conf_mat)
#print(classification_report(sentList, y_pred, target_names=target_names))
#clf_rf.fit(df_full, sentList)
#plot_confusion_matrix(clf_rf, df_full, sentList, normalize= 'true')



#Deep Learning Section 
tokenizer = Tokenizer(num_words=12500)
tokenizer.fit_on_texts(PhraseList)
vocab_size = len(tokenizer.word_index) + 1
encoded_docs = tokenizer.texts_to_sequences(PhraseList)
padded_sequence = pad_sequences(encoded_docs, maxlen=200)


# Build the model
embedding_vector_length = 32

#Model 5
model = Sequential()
model.add(Embedding(vocab_size, embedding_vector_length,input_length=200) )
model.add(SpatialDropout1D(0.15))
model.add(LSTM(50, dropout=0.25, recurrent_dropout=0.25))
#model.add(Dropout(0.2))
model.add(Dense(5, activation='sigmoid'))
model.compile(loss='SparseCategoricalCrossentropy',optimizer='adam', metrics=['accuracy'])
print(model.summary())


history = model.fit(padded_sequence, sentList, validation_split=0.2, epochs=5, batch_size=32)

print(history)


































#More advanced LSTM 
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(PhraseList)
vocab_size = len(tokenizer.word_index) + 1
encoded_docs = tokenizer.texts_to_sequences(PhraseList)
padded_sequence = pad_sequences(encoded_docs, maxlen=200)


# Build the model
embedding_vector_length = 32

#Model 6 10mins per epoch
model = Sequential()
model.add(Embedding(vocab_size, embedding_vector_length,input_length=200) )
model.add(SpatialDropout1D(0.15))
model.add(LSTM(75, dropout=0.25, recurrent_dropout=0.25))
model.add(Dropout(0.15))
model.add(Dense(5, activation='sigmoid'))
model.compile(loss='SparseCategoricalCrossentropy',optimizer='adam', metrics=['accuracy'])
print(model.summary())


#history = model.fit(padded_sequence, sentList, validation_split=0.2, epochs=5, batch_size=32)






#Model 7 25mins per epoch
model = Sequential()
model.add(Embedding(vocab_size, embedding_vector_length,input_length=200) )
#model.add(SpatialDropout1D(0.15))
model.add(LSTM(25, dropout=0.25, recurrent_dropout=0.25, return_sequences=True))
#model.add(Dropout(0.15))
model.add(LSTM(12, dropout=0.25, recurrent_dropout=0.25, return_sequences=True))
#model.add(Dropout(0.15))
model.add(LSTM(6, dropout=0.25, recurrent_dropout=0.25))
#model.add(Dropout(0.15))
model.add(Dense(5, activation='sigmoid'))
model.compile(loss='SparseCategoricalCrossentropy',optimizer='adam', metrics=['accuracy'])
print(model.summary())


#history = model.fit(padded_sequence, sentList, validation_split=0.2, epochs=5, batch_size=32)
plt.show()
