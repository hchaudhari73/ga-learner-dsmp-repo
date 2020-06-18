# --------------
import pandas as pd 
news = pd.read_csv(path)
news = news[["TITLE", "CATEGORY"]]
dist = news.CATEGORY.value_counts()
print(dist)
print(news.head())


# --------------
import re
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

news["TITLE"] = news["TITLE"].apply(lambda x: re.sub("[^a-zA-Z]", " ",x))
news["TITLE"] = news["TITLE"].apply(lambda x: x.lower().split())
stop = stopwords.words("english")
news["TITLE"] = news["TITLE"].apply(lambda x: [i for i in x if i not in stop])
news["TITLE"] = news["TITLE"].apply(lambda x: " ".join(x))
X_train, X_test, Y_train, Y_test = train_test_split(news["TITLE"], news["CATEGORY"], test_size=0.2, random_state=3)


# --------------
# Code starts here
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# initialize count vectorizer
count_vectorizer = CountVectorizer()
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,3))
# initialize tfidf vectorizer
X_train_count = count_vectorizer.fit_transform(X_train)
X_test_count = count_vectorizer.transform(X_test)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)





# --------------
# Code starts here
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import numpy as np
# initialize multinomial naive bayes
nb_1 = MultinomialNB()
nb_2 = MultinomialNB()

# fit on count vectorizer training data
nb_1.fit(X_train_count, Y_train)
y_pred_count = nb_1.predict(X_test_count)
# fit on tfidf vectorizer training data
nb_2.fit(X_train_tfidf, Y_train)
y_pred_tfidf = nb_2.predict(X_test_tfidf)
# accuracy with count vectorizer
acc_count_nb = np.round(accuracy_score(y_pred_count, Y_test),2)

# accuracy with tfidf vectorizer
acc_tfidf_nb = np.round(accuracy_score(y_pred_tfidf, Y_test),2)

# display accuracies
print(acc_count_nb)
print(acc_tfidf_nb)
# Code ends here


# --------------
import warnings
warnings.filterwarnings('ignore')
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
# initialize logistic regression
logreg_1 = OneVsRestClassifier(LogisticRegression(random_state=10))
logreg_2 = OneVsRestClassifier(LogisticRegression(random_state=10))
# fit on count vectorizer training data
logreg_1.fit(X_train_count, Y_train)
y_pred_count = logreg_1.predict(X_test_count)
# fit on tfidf vectorizer training data
logreg_2.fit(X_train_tfidf, Y_train)
y_pred_tfidf = logreg_2.predict(X_test_tfidf)
# accuracy with count vectorizer
acc_count_logreg = accuracy_score(Y_test, y_pred_count)
# accuracy with tfidf vectorizer
acc_tfidf_logreg = accuracy_score(Y_test, y_pred_tfidf)

# display accuracies
print(acc_count_logreg, '\n',acc_tfidf_logreg)

# Code ends here


