# --------------
import pandas as pd
from sklearn import preprocessing

#path : File path

# Code starts here


# read the dataset
dataset = pd.read_csv(path)


# look at the first five columns
print(dataset.head())

# Check if there's any column which is not useful and remove it like the column id
dataset.drop("Id",axis = 1, inplace = True)

# check the statistical description
dataset.describe()


# --------------
# We will visualize all the attributes using Violin Plot - a combination of box and density plots
import seaborn as sns
from matplotlib import pyplot as plt

#names of all the attributes 
cols = dataset.columns

#number of attributes (exclude target)
size = len(cols) - 1 

#x-axis has target attribute to distinguish between classes
x = dataset[cols[-1]]

#y-axis shows values of an attribute
y = dataset.drop(cols[-1], axis = 1)

#Plot violin for all attributes
for i in range(size):
    sns.violinplot(x = dataset[cols[i]])
    plt.show()


# --------------
import numpy
upper_threshold = 0.5
lower_threshold = -0.5


# Code Starts Here
subset_train = pd.DataFrame()
for col in cols[:10]:
    subset_train[col] = dataset[col]

data_corr = subset_train.corr()

sns.heatmap(data_corr, annot=True)
plt.show()

correlation = data_corr.unstack().sort_values(kind='quicksort')
corr_var_list = correlation[(abs(correlation)>0.5) & (abs(correlation)<1)]

# Code ends here




# --------------
#Import libraries 
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
# Identify the unnecessary columns and remove it 
dataset.drop(columns=['Soil_Type7', 'Soil_Type15'], inplace=True)

X = dataset.drop(cols[-1], axis = 1)
y = dataset[cols[-1]]

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 0, test_size = 0.2)

scaler = StandardScaler()

X_train_temp = scaler.fit_transform(X_train.iloc[:,0:size])
X_train1 = np.concatenate((X_train_temp, X_train.iloc[:,size:]),axis = 1)

X_test_temp = scaler.fit_transform(X_test.iloc[:,0:size])
X_test1 = np.concatenate((X_test_temp, X_test.iloc[:,size:]),axis = 1)

scaled_features_train_df = pd.DataFrame(X_train1, index=X_train.index, columns=X_train.columns)
scaled_features_test_df = pd.DataFrame(X_test1, index=X_test.index, columns=X_test.columns)



# --------------
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif


# Write your solution here:
skb = SelectPercentile(score_func = f_classif, percentile = 90)
predictors = skb.fit_transform(X_train1, y_train)
scores = skb.scores_
Features = X_train.columns
dataframe = pd.DataFrame({"Features":Features, "scores": scores})
sorted_df = dataframe.sort_values(by = "scores", ascending=False)
top_k_predictors = list(sorted_df['Features'][:predictors.shape[1]])

print(top_k_predictors)


# --------------
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score
clf = OneVsRestClassifier(LogisticRegression())
clf1 = OneVsRestClassifier(LogisticRegression())

model_fit_all_features = clf1.fit(X_train, Y_train)

predictions_all_features = model_fit_all_features.predict(X_test)

score_all_features = accuracy_score(Y_test, predictions_all_features)

print(score_all_features)

model_fit_top_features = clf.fit(scaled_features_train_df[top_k_predictors], Y_train)

predictions_top_features = model_fit_top_features.predict(scaled_features_test_df[top_k_predictors])

score_top_features = accuracy_score(Y_test, predictions_top_features)

print(score_top_features)


