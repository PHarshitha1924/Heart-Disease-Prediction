from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import pandas as pd
import warnings

warnings.filterwarnings("ignore",category=UserWarning, message="Found Unknown categories in columns.*")
df=pd.read_csv("heart.csv")
X=df.drop('target',axis=1)
Y=df['target']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
numerical_cols=X.select_dtypes(include=['int64','float64']).columns
categorical_cols=X.select_dtypes(include=['category','object']).columns
model=RandomForestClassifier(n_estimators=100,random_state=42)
preprocessor=ColumnTransformer([('cat',OneHotEncoder(drop='first',handle_unknown='ignore'),categorical_cols),('num',StandardScaler(),numerical_cols)])
pipeline=Pipeline([('preprocessing',preprocessor),('training',model)])
pipeline.fit(X_train,Y_train)
Y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("Precision:", precision_score(Y_test, Y_pred))
print("Recall:", recall_score(Y_test, Y_pred))
print("F1 Score:", f1_score(Y_test, Y_pred))

print("\nClassification Report:\n", classification_report(Y_test, Y_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred))

print("\n--- Heart Disease Prediction ---")

new_data = {}
for col in X.columns:
    value = input(f"Enter {col}: ")
    try:
        new_data[col] = float(value)
    except:
        new_data[col] = value

new_df = pd.DataFrame([new_data])

prediction = pipeline.predict(new_df)[0]

if prediction == 1:
    print("\nPrediction: HEART DISEASE (1)")
else:
    print("\nPrediction: NO HEART DISEASE (0)")