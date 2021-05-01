import pandas as pd
import gzip
import json
import numpy as np

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

df = getDF('Gift_Cards_5.json.gz')
df = df[df['reviewText'].notna()]

from spell_check import fixSentence
df['reviewText'] = df['reviewText'].apply(lambda x: fixSentence(x))

print('CHECKPOINT: Spell Check Complete')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

depth = [int(x) for x in np.linspace(10, 40, num = 5)]
estimators = [int(x) for x in np.linspace(start = 50, stop = 200, num = 5)]

rf = RandomForestClassifier()
grid = {'max_depth': depth, 'n_estimators': estimators}
GS_object = GridSearchCV(estimator = rf, param_grid = grid)

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer

token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts = cv.fit_transform(df['reviewText'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(text_counts, df['overall'], test_size = 0.25, random_state = 5)

GS_object.fit(X_train, y_train)

from sklearn import metrics
#train scores
train_predicted = GS_object.predict(X_train)
train_accuracy_score = metrics.accuracy_score(y_train, train_predicted)
train_f1_score = metrics.f1_score(y_train, train_predicted, average = 'macro')
print(str('{:04.2f}'.format(train_accuracy_score*100))+'%')
print(train_f1_score)

#test scores
test_predicted = GS_object.predict(X_test)
test_accuracy_score = metrics.accuracy_score(y_test, test_predicted)
test_f1_score = metrics.f1_score(y_test, test_predicted, average = 'macro')
print(str('{:04.2f}'.format(train_accuracy_score*100))+'%')
print(test_f1_score)
