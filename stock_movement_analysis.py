#!/usr/bin/env python
# coding: utf-8

# # Stock Movement Analysis

# In[39]:


# Importing required libraries
import praw
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# In[40]:


# Reddit API credentials
reddit = praw.Reddit(
    client_id='wLMcSW8KYWa3WtT8Eqcruw',
    client_secret='TB26qbgfYxtlBQNAXRsYemIxwbavhw',
    user_agent='Stock Movement Analysis',
    username='Puzzleheaded_22514',
    password='Sakshi@22514'
)


# In[41]:


print(reddit.read_only) # Should print False if connected successfull


# In[42]:


# Scraping data from r/stocks
def scrape_reddit(subreddit_name, limit=500):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for post in subreddit.hot(limit=limit):
        posts.append({
            'title': post.title,
            'selftext': post.selftext,
            'created_utc': post.created_utc
        })
    return pd.DataFrame(posts)


# In[43]:


data = scrape_reddit('stocks')
data.head()


# In[44]:


# Combining title and selftext for analysis
data['content'] = data['title'] + ' ' + data['selftext']
data.drop(['title', 'selftext'], axis=1, inplace=True)


# In[45]:


data = data.dropna(subset=['content'])


# In[46]:


# Converting timestamp to datetime
data['date'] = pd.to_datetime(data['created_utc'], unit='s')
data.drop(['created_utc'], axis=1, inplace=True)


# In[47]:


data.head()


# In[48]:


nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()


# In[49]:


data['sentiment'] = data['content'].apply(lambda x: sid.polarity_scores(x)['compound'])


# In[50]:


data['sentiment_label'] = data['sentiment'].apply(
    lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral')
)


# In[51]:


data.head()


# In[52]:


# Extracting relevant features
data['word_count'] = data['content'].apply(lambda x: len(x.split()))
data['char_count'] = data['content'].apply(lambda x: len(x))


# In[53]:


# Preparing data for modeling
X = data[['sentiment', 'word_count', 'char_count']]
y = data['sentiment_label']
y = y.map({'positive': 1, 'negative': -1, 'neutral': 0})


# In[54]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[55]:


# Training Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


# In[56]:


y_pred = model.predict(X_test)


# In[57]:


# Evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[ ]:


# Sentiment distribution
sns.countplot(x=data['sentiment_label'], palette='viridis')
plt.title('Sentiment Distribution')
plt.show()


# In[ ]:




