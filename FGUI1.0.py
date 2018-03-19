import json
import re
from string import punctuation
import tweepy
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from multiprocessing import Pool, cpu_count
import numpy as np
from flask import Flask, render_template, request, redirect
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



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('/index.html')

if __name__ == '__main__':

    @app.route('/search', methods=['POST', 'GET'])
    def search():
        if request.method == 'POST':
            input=request.form.to_dict(flat=False)
            print(request.form.to_dict(flat=False))

            def createDataSet(in_string):  # function to build the test and training dataset
                # this function searches for a keyword in a set of tweets (local database)
                # it returns tweets that contain the search keyword (test dataset)

                file_name = "tweets.json"  # json file from which tweets are retrieved
                temp_string = []  # used to hold the input string temporarily

                # make sure the input string comprises of characters
                for i in in_string:
                    temp_string.append(i)
                search_term = ''.join(temp_string[1:len(temp_string) - 1])

                count = 0  # fetched tweets counter
                with open(file_name, 'r') as f:
                    print("Searching for tweets...")
                    for tweet in f:
                        t = json.loads(tweet)
                        if search_term in t["text"]:
                            testData.append({"text": t["text"], "label": "neutral"})
                            count += 1

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



            testData = []

            search_string = input['search']  # prompt user for search keyword
            createDataSet(search_string)  # build test and training dataset

            tweetProcessor = PreProcessTweets()

            print("Preprocessing" + str(len(testData)) + "Test Data...")
            ppTestData = tweetProcessor.processTweets(testData)

            classifier_f = open("naivebayes.pickle", "rb")
            NBayesClassifier = pickle.load(classifier_f)
            classifier_f.close()
            print(NBayesClassifier)
            pickle_out = open("dict.pickle", "rb")
            word_features = pickle.load(pickle_out).keys()
            pickle_out.close()
            print(word_features)


            # run the classifier on the preprocessed test dataset
            print("Acquiring Results...")
            NBResultLabels = [NBayesClassifier.classify(extract_features(tweet[0])) for tweet in ppTestData]

            # print(NBResultLabels)

            # get the majority vote and print the sentiment
            if NBResultLabels.count('positive') > NBResultLabels.count('negative'):
                print("NB Result Positive Sentiment " + str(
                    100 * NBResultLabels.count('positive') / len(NBResultLabels)) + "%")
            else:
                print("NB Result Negative Sentiment " + str(
                    100 * NBResultLabels.count('negative') / len(NBResultLabels)) + "%")

            # print output
            i = 0
            with open('test1.json', 'a') as f:
                while i < len(testData):
                    print("Polarity: " + json.dumps(NBResultLabels[i]) + " || Text: " + json.dumps(testData[i]["text"]))
                    data = {
                        'text': json.dumps(testData[i]["text"]),
                        'Polarity': json.dumps(NBResultLabels[i])
                    }
                    json.dump(data, f)
                    f.write('\n')
                    i += 1
            a = "hello"
        return a


    app.run(debug=True)