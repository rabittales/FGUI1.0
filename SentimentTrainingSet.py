import json
import re
from string import punctuation
import tweepy
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from multiprocessing import Pool, cpu_count
import numpy as np
from flask import Flask, render_template
from tqdm import tqdm


def _processT(tweets):
    outputs = []
    for x in tqdm(tweets, miniters=10):
        tweet = x['text']
        # convert to lower case
        tweet = tweet.lower()
        # replace links with the word 'URL'
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)
        # replace @username with 'AT_USER'
        tweet = re.sub('@[^\s]+', 'AT_USER', tweet)
        # replace #word with 'word'
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
        # tokenize the tweet into a list of words
        tweet = word_tokenize(tweet)
        outputs.append(([word for word in tweet if
                        word not in set(stopwords.words('english') + list(punctuation) + ['AT_USER', 'URL'])], x["label"]))
    return outputs

def createDataSet():  # function to build the test and training dataset
    # this function searches for a keyword in a set of tweets (local database)
    # it returns tweets that contain the search keyword (test dataset)

    file_name = "tweets.json"  # json file from which tweets are retrieved
    temp_string = []  # used to hold the input string temporarily

    # make sure the input string comprises of characters

    count = 0  # fetched tweets counter
    with open(file_name, 'r') as f:
        print("Searching for tweets...")
        for tweet in f:
                count += 1
                t = json.loads(tweet)
                if t["label"] == "0":
                    trainingData.append({"tweet_id": t["tweet_id"], "label": "negative", "text": t["text"]})
                else:
                    trainingData.append({"tweet_id": t["tweet_id"], "label": "positive", "text": t["text"]})

    print(str(count) + " Tweets found.\n")

class PreProcessTweets:
    # this class preprocesses all the tweets, both test and training
    # it uses regular expressions and Natural Language Toolkit

    def __init__(self):
        self._stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER', 'URL'])
    # END OF FUNCTION

    def processTweets(self, list_of_tweets):
        # the following is a list of dictionaries which have the keys "text" and "label"
        # this list is a list of tuples. Each tuple is a tweet (list of words) and its label
        p = Pool(cpu_count())
        chunk = [x.tolist() for x in np.array_split(list_of_tweets, cpu_count())]
        processedTweets = np.concatenate([x for x in p.map(_processT, chunk) if len(x) > 0]).tolist()
        p.close()
        p.join()
        #for tweet in list_of_tweets:
        #    processedTweets.append((self._processTweet(tweet["text"]), tweet["label"]))
        return processedTweets
    # END OF FUNCTION
# END OF CLASS

def buildVocabulary(ppTrainingData):
    # the following will give a list in which all the words in all the tweets are present
    # these have to be de-duped. Each word occurs in this list as many times as it appears in the corpus
    all_words = []
    for (words, sentiment) in ppTrainingData:
        all_words.extend(words)

    wordlist = nltk.FreqDist(all_words)  # create a dictionary with each word and its frequency
    word_features = wordlist

    # return the unique list of words in the corpus
    return word_features
# END OF FUNCTION

def extract_features(tweet):
    # function to take each tweet in the training data and represent it with
    # the presence or absence of a word in the vocabulary

    tweet_words = set(tweet)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in tweet_words)
        # this gives a dictionary with keys like 'contains word' and values as True or False
    return features
# END OF FUNCTION

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':

    trainingData = []
    createDataSet()
    tweetProcessor = PreProcessTweets()
    print("Preprocessing " + str(len(trainingData)) + " Training Data...")
    ppTrainingData = tweetProcessor.processTweets(trainingData)


    # extract the features and train the classifier
    print("Building vocabulary for Training Data...")
    word_features = buildVocabulary(ppTrainingData)
    print(word_features)
    pickle_out = open("dict.pickle", "wb")
    pickle.dump(word_features, pickle_out)
    pickle_out.close()

    print("Classifying features...")
    trainingFeatures = nltk.classify.apply_features(extract_features, ppTrainingData)

    # train using Naive Bayes
    print("Running Data on Naive Bayes Classifier...")
    NBayesClassifier = nltk.NaiveBayesClassifier.train(trainingFeatures)

    save_classifier = open("naivebayes.pickle", "wb")
    pickle.dump(NBayesClassifier, save_classifier)
    save_classifier.close()

