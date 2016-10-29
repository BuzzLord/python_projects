import sys
import codecs
import tweepy
from twitter_access import TwitterAccess
from textblob import TextBlob

sys.stdout = codecs.getwriter('cp850')(sys.stdout.buffer, 'strict') 

access = TwitterAccess()
api = access.getTweepyApi()

public_tweets = api.search('Sirajology')

for tweet in public_tweets:
    try:
        print(tweet.text)
        analysis = TextBlob(tweet.text)
        print(analysis.sentiment)
        print("\n")
    except:
        print("Exception, skipping...\n")

