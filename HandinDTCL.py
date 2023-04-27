#import libraries
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
import numpy as np
import time
from sklearn.linear_model import LogisticRegression

#Read in the data from the file
df=pd.read_csv('stroke_data.csv')

#Visualize the data
print(df.head())

#replace NaN values with the mean of the column
df['bmi'].fillna(df['bmi'].mean(),inplace=True)
df['age'].fillna(df['bmi'].mean(),inplace=True)
df['avg_glucose_level'].fillna(df['bmi'].mean(),inplace=True)
df.fillna(0, inplace=True)
#normalize the data
df['bmi']=(df['bmi']-df['bmi'].min())/(df['bmi'].max()-df['bmi'].min())
df['age']=(df['age']-df['age'].min())/(df['age'].max()-df['age'].min())
df['avg_glucose_level']=(df['avg_glucose_level']-df['avg_glucose_level'].min())/(df['avg_glucose_level'].max()-df['avg_glucose_level'].min())
#Visualize the data
print(df.head())

X=df.drop('stroke',axis=1)
y=df['stroke']

#Split the data into training and testing data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#############################################################################################################################################
#Decision Tree
#############################################################################################################################################
start_time = time.time()
#Create the decision tree model
dt=DecisionTreeClassifier(criterion='entropy', max_features='log2',random_state=42)

#Fit the model to the training data
dt.fit(X_train,y_train)

#Predict the test data
y_pred=dt.predict(X_test)

dt_acc=accuracy_score(y_test,y_pred)
#Calculate the accuracy of the model
print('Accuracy of Decision Tree: ',dt_acc)

end_time = time.time()
execution_time = end_time - start_time
print('Execution time of Decision Tree: ',execution_time, "seconds")
#############################################################################################################################################
#Ensemble Methods
#############################################################################################################################################

# Create and fit the models
#AdaBoost
start_time = time.time()
ada = AdaBoostClassifier(n_estimators=100)
ada.fit(X_train, y_train)
end_time = time.time()
execution_time1= end_time - start_time
print('Execution time of AdaBoost: ',execution_time1, "seconds")
#XGBoost
start_time = time.time()
xgb = XGBClassifier(n_estimators=100)
xgb.fit(X_train, y_train)
end_time = time.time()
execution_time2 = end_time - start_time
print('Execution time of XGBoost: ',execution_time2, "seconds")
#Bagging
start_time = time.time()
bag = BaggingClassifier(n_estimators=100)
bag.fit(X_train, y_train)
end_time = time.time()
execution_time3 = end_time - start_time
print('Execution time of Bagging: ',execution_time3, "seconds")
#Random Forest
start_time = time.time()
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
end_time = time.time()
execution_time4 = end_time - start_time
print('Execution time of Random Forest: ',execution_time4, "seconds")

# Predict using the models
ada_pred = ada.predict(X_test)
xgb_pred = xgb.predict(X_test)
bag_pred = bag.predict(X_test)
rf_pred = rf.predict(X_test)

# Calculate the accuracies of each model
ada_acc = accuracy_score(y_test, ada_pred)
xgb_acc = accuracy_score(y_test, xgb_pred)
bag_acc = accuracy_score(y_test, bag_pred)
rf_acc = accuracy_score(y_test, rf_pred)

print('AdaBoost accuracy:', ada_acc)
print('XGBoost accuracy:', xgb_acc)
print('Bagging accuracy:', bag_acc)
print('Random Forest accuracy:', rf_acc)

#Hard voting
from sklearn.ensemble import VotingClassifier
start_time = time.time()
hard_voting_clf = VotingClassifier(estimators=[('ada', ada),('bag', bag), ('rf', rf),('xgb',xgb)], voting='hard')
hard_voting_clf.fit(X_train, y_train)
hard_voting_pred = hard_voting_clf.predict(X_test)
voting_acc = accuracy_score(y_test, hard_voting_pred)
print('Hard Voting accuracy:', voting_acc)
end_time = time.time()
execution_time5 = end_time - start_time
print('Execution time of Hard Voting: ',execution_time5, "seconds")

#Soft voting
start_time = time.time()
voting_clf = VotingClassifier(estimators=[('ada', ada),('bag', bag), ('rf', rf),('xgb',xgb)], voting='soft')
voting_clf.fit(X_train, y_train)
voting_pred = voting_clf.predict(X_test)
voting_acc = accuracy_score(y_test, voting_pred)
print(' Soft Voting accuracy:', voting_acc)
end_time = time.time()
execution_time6 = end_time - start_time
print('Execution time of Soft Voting: ',execution_time6, "seconds")

#############################################################################################################################################
#Neural Network with mlp
#############################################################################################################################################
start_time = time.time()
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10,10), max_iter=100)
mlp.fit(X_train, y_train)
mlp_pred = mlp.predict(X_test)
mlp_acc = accuracy_score(y_test, mlp_pred)
print('Neural Network accuracy:', mlp_acc)
end_time = time.time()
execution_time7 = end_time - start_time
print('Execution time of Neural Network: ',execution_time7, "seconds")
#############################################################################################################################################
##LOGISTIC REGRESSION
#############################################################################################################################################
#Create the logistic regression model
start_time = time.time()
lr=LogisticRegression(random_state=42)

#Fit the model to the training data
lr.fit(X_train,y_train)

#Predict the test data
y_pred=lr.predict(X_test)

#Calculate the accuracy of the model
lr_acc=accuracy_score(y_test,y_pred)
end_time = time.time()
execution_time8 = end_time - start_time
print('Execution time of Logistic Regression: ',execution_time8, "seconds")
print('Accuracy of Logistic Regression: ',lr_acc)

#############################################################################################################################################
#Plot the accuracies of each model and the execution time
#############################################################################################################################################
#plot execution time of each model vs accuracy of each model
plt.figure(figsize=(20,10))
plt.bar(['Decision Tree','AdaBoost','XGBoost','Bagging','Random Forest','Hard Voting','Soft Voting','Neural Network','Logistic Regression'],[execution_time,execution_time1,execution_time2,execution_time3,execution_time4,execution_time5,execution_time6,execution_time7,execution_time8])
plt.bar(['Decision Tree','AdaBoost','XGBoost','Bagging','Random Forest','Hard Voting','Soft Voting','Neural Network','Logistic Regression'],[dt_acc*100,ada_acc*100,xgb_acc*100,bag_acc*100,rf_acc*100,voting_acc*100,voting_acc*100,mlp_acc*100,lr_acc*100], alpha=0.2)
plt.show()
#############################################################################################################################################

#User input
while True:
    print('Enter the following information to predict if you will have a stroke:')
    sex=input("Are you a female(0) or a male(1)")
    age=input("What is your age?")
    hypertension=input("Do you have hypertension(0 for no, 1 for yes)?")
    heart_disease=input("Do you have heart disease(0 for no, 1 for yes)?")
    ever_married=input("Are you married(0 for no, 1 for yes)?")
    work_type=input("What is your work type(0 for children, 1 for Govt_job, 2 for Never_worked, 3 for Private, 4 for Self-employed)?")
    Residence_type=input("What is your residence type(0 for Rural, 1 for Urban)?")
    avg_glucose_level=input("What is your average glucose level?")
    bmi=input("What is your bmi?")
    smoking_status=input("What is your smoking status(0 for formerly smoked, 1 for never smoked, 2 for smokes)?")

    #make the prediction
    UI=np.array([sex,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status])
    UI=UI.astype('float64')
    #reshape the array
    UI=np.array(UI).reshape(1,-1)
    
    #print the prediction
    print('Decision Tree prediction: ',dt.predict(UI))
    print('XGBoost prediction: ',xgb.predict(UI))
    print('AdaBoost prediction: ',ada.predict(UI))
    print('Bagging prediction: ',bag.predict(UI))
    print('Random Forest prediction: ',rf.predict(UI))
    print('Hard Voting prediction: ',hard_voting_clf.predict(UI))
    print('Soft Voting prediction: ',voting_clf.predict(UI))
    print('Neural Network prediction: ',mlp.predict(UI))
    print('Logistic Regression prediction: ',lr.predict(UI))
    
    #ask if the user wants to continue
    if input('Do you want to continue? (y/n)') == 'n':
        break
    else:
        continue
  
    