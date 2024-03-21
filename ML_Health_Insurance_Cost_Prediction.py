import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn import metrics

import joblib

data = pd.read_csv("insurance.csv")

print("Rows",data.shape[0])
print("Columns",data.shape[1])

data.sex = data.sex.map({"female": 0, "male": 1})
data.smoker = data.smoker.map({"yes": 1, "no": 0})
data.region = data.region.map({
    "southwest": 1,
    "southeast": 2,
    "northwest": 3,
    "northeast": 4
})

X = data.drop(["charges"], axis=1)
y = data["charges"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

lr = LinearRegression()
lr.fit(X_train,y_train)

svm = SVR()
svm.fit(X_train,y_train)

rf = RandomForestRegressor()
rf.fit(X_train,y_train)

gr = GradientBoostingRegressor()
gr.fit(X_train,y_train)

y_pred1 = lr.predict(X_test)
y_pred2 = svm.predict(X_test)
y_pred3 = rf.predict(X_test)
y_pred4 = gr.predict(X_test)

data_pred = pd.DataFrame({
    'Actual': y_test,
    'Lr': y_pred1,
    'svm': y_pred2,
    'rf': y_pred3,
    'gr': y_pred4
})

plt.figure(figsize=(12, 8))

plt.subplot(221)
plt.plot(data_pred['Actual'].iloc[0:11], label='Actual', color='blue', marker='o')
plt.plot(data_pred['Lr'].iloc[0:11], label="Lr", color='orange', marker='x')
plt.title('Linear Regression Prediction')
plt.xlabel('Data Point')
plt.ylabel('Value')
plt.legend()

plt.subplot(222)
plt.plot(data_pred['Actual'].iloc[0:11], label='Actual', color='blue', marker='o')
plt.plot(data_pred['svm'].iloc[0:11], label="svr", color='green', marker='x')
plt.title('Support Vector Machine Prediction')
plt.xlabel('Data Point')
plt.ylabel('Value')
plt.legend()

plt.subplot(223)
plt.plot(data_pred['Actual'].iloc[0:11], label='Actual', color='blue', marker='o')
plt.plot(data_pred['rf'].iloc[0:11], label="rf", color='red', marker='x')
plt.title('Random Forest Prediction')
plt.xlabel('Data Point')
plt.ylabel('Value')
plt.legend()

plt.subplot(224)
plt.plot(data_pred['Actual'].iloc[0:11], label='Actual', color='blue', marker='o')
plt.plot(data_pred['gr'].iloc[0:11], label="gr", color='purple', marker='x')
plt.title('Gradient Boosting Prediction')
plt.xlabel('Data Point')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()

plt.show()

score1 = metrics.r2_score(y_test,y_pred1)
score2 = metrics.r2_score(y_test,y_pred2)
score3 = metrics.r2_score(y_test,y_pred3)
score4 = metrics.r2_score(y_test,y_pred4)

print(score1,score2,score3,score4)

s1 = metrics.mean_absolute_error(y_test,y_pred1)
s2 = metrics.mean_absolute_error(y_test,y_pred2)
s3 = metrics.mean_absolute_error(y_test,y_pred3)
s4 = metrics.mean_absolute_error(y_test,y_pred4)

print(s1,s2,s3,s4)

data_predict = pd.DataFrame(data={
    'age' : 40,
    'sex' : 0,
    'bmi' : 40.30,
    'children' : 4,
    'smoker' : 0,
    'region' : 3
}, index=["v"])

new_pred = gr.predict(data_predict)
print("Medical Insurance cost for New Customer is : ",new_pred[0])

gr = GradientBoostingRegressor()
gr.fit(X,y)

new_pred = gr.predict(data_predict)
print("Medical Insurance cost for New Customer is : ",new_pred[0])


joblib.dump(gr, "model_")

gr_test = joblib.load("model_")

new_pred_test = gr_test.predict(data_predict)
print("Medical Insurance cost for New Customer is : ",new_pred_test[0])
