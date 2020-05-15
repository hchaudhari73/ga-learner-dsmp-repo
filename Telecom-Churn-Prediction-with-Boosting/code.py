# --------------
import pandas as pd
from sklearn.model_selection import train_test_split
#path - Path of file 

# Code starts here
df = pd.read_csv(path)
X = df.drop(columns=['customerID','Churn'])
y = df.Churn

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)



# --------------
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Code starts here
X_train.replace(" ", np.NaN, inplace = True)
X_test.replace(" ", np.NaN, inplace = True)

X_train["TotalCharges"] = X_train["TotalCharges"].astype("float")
X_test["TotalCharges"] = X_test["TotalCharges"].astype("float")

X_train.TotalCharges.fillna(X_train.TotalCharges.mean(), inplace=True)
X_test.TotalCharges.fillna(X_test.TotalCharges.mean(), inplace=True)

print(X_train.isnull().sum())

cat_col = ["gender", "Partner",	"Dependents",	"tenure",	"PhoneService",	"MultipleLines",	"InternetService",	"OnlineSecurity",	"OnlineBackup",	"DeviceProtection",	"TechSupport",	"StreamingTV",	"StreamingMovies",	"Contract",	"PaperlessBilling",	"PaymentMethod"]

for col in cat_col:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    le = LabelEncoder()
    X_test[col] = le.fit_transform(X_test[col])
y_train = y_train.map(lambda x:1 if x=="Yes" else 0)
y_test = y_test.map(lambda x:1 if x=="Yes" else 0)



# --------------
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# Code starts here
print(X_train.head())
print("-"*30)
print(X_test.head())
print("-"*30)
print(y_train.head())
print("-"*30)
print(y_test.head())
print("-"*30)

ada_model = AdaBoostClassifier(random_state=0)
ada_model.fit(X_train, y_train)
y_pred = ada_model.predict(X_test)
ada_score = accuracy_score(y_test, y_pred)

ada_cm = confusion_matrix(y_pred, y_test)

ada_cr = classification_report(y_pred, y_test)




# --------------
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

#Parameter list
parameters={'learning_rate':[0.1,0.15,0.2,0.25,0.3],
            'max_depth':range(1,3)}

# Code starts here
xgb_model = XGBClassifier(random_state = 0)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
xgb_score = accuracy_score(y_pred, y_test)
xgb_cm = confusion_matrix(y_pred, y_test)
xgb_cr = classification_report(y_pred, y_test)

xgb_clf = XGBClassifier(random_state=0)
clf_model = GridSearchCV(estimator=xgb_clf, param_grid=parameters)
clf_model.fit(X_train, y_train)
clf_score = accuracy_score(y_pred, y_test)
clf_cm = confusion_matrix(y_pred, y_test)
clf_cr = classification_report(y_pred, y_test)








