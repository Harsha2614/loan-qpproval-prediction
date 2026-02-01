import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report ,  confusion_matrix
import joblib
#import
data=pd.read_csv("data/loan.csv")

#preprocess
data=data.dropna()

X=data.drop("Loan_Status",axis=1)
Y = data["Loan_Status"].map({"Y": 1, "N": 0})

#encode
X=pd.get_dummies(X)

#split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

#train
model=RandomForestClassifier()
model.fit(X_train,Y_train)

#evaluate
pred=model.predict(X_test)
acc=accuracy_score(Y_test,pred)
clasification=classification_report(Y_test,pred)
print("Accuracy",acc)
print(clasification)

print(pred)

#save
joblib.dump(model,"model/loan_model.pkl")
joblib.dump(X.columns,"model/features.pkl")