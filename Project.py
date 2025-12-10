#Name: Shivam Patel
#Date: December 12, 2025
#Description: This project predicts if the titanic passengers survived or not using machine learning algorithms.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

#--- Step 1: Load the dataset ---#
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

print("Step 1..........................................Complete")

#--- Step 2: Data cleaning ---#

#checking for missing values
data['Age'] = data['Age'].fillna(data['Age'].mean())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

#dropping unnecessary columns
data.drop(columns=['Cabin', 'Ticket', 'Name'], inplace=True)

#Encode categorical columns
data['Sex'] = data['Sex'].map({'male': 1, 'female': 0})
data = pd.get_dummies(data, columns=['Embarked'], drop_first=False)
print("Step 2..........................................Complete")

#--- Step 3: Feature selection and encoding ---#
target = 'Survived'
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']

x = data[features]
y = data[target]

#convert cronical features to numerical
x = pd.get_dummies(x, drop_first=True)

print("Step 3..........................................Complete")

#--- Step 4: Split the dataset ---#
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("Step 4..........................................Complete")

#--- Step 5: Scale the data ---#
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)

print("Step 5..........................................Complete")

#--- Step 6: Train and evaluate multiple models ---#
model = {
    'Logistic Regression' : [
        LogisticRegression(max_iter=200, C=0.1),
        LogisticRegression(max_iter=200, C=1.0),
        LogisticRegression(max_iter=200, C=10.0)
    ],
    'KNN' : [
        KNeighborsClassifier(n_neighbors=3, weights='uniform', metric='euclidean'),
        KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean'),
        KNeighborsClassifier(n_neighbors=7, weights='uniform', metric='manhattan')
    ],
    'Neural Network' : [
        MLPClassifier(hidden_layer_sizes=(8,), max_iter=1000, random_state=42),
        MLPClassifier(hidden_layer_sizes=(16,8), max_iter=1000, random_state=42),
        MLPClassifier(hidden_layer_sizes=(32,16,8), activation='tanh', max_iter=1000, random_state=42)
    ]
}

result = {}
best_models = {}

for model_name, model_list in model.items():
    best_accuracy = 0
    best_model = None
    best_params = None

    for m in model_list:
        m.fit(x_train_scaled, y_train)
        y_pred = m.predict(x_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = m
            best_params = m.get_params()
            best_report = classification_report(y_test, y_pred)
            best_conf = confusion_matrix(y_test, y_pred)
        
    best_models[model_name] = best_model
    result[model_name] = {
        'accuracy': best_accuracy,
        'best_params': best_params,
        'report': best_report,
        'Confusion_matrix': best_conf
    }
print("Step 6..........................................Complete")

print("-------------------Project 1 Complete-------------------")

#--- printing detailed report of each model ---#
print("========================================================")
print("================ Detailed Model Reports ================")
for name, metrics in result.items():
    print(f"\nModel: {name}")
    if "Logistic Regression" in name:
        print(f"Best Hyperparameter: C = {metrics['best_params']['C']}")
    elif "KNN" in name:
        print(f"Best Hyperparameter: n_neighbors = {metrics['best_params']['n_neighbors']}")
    elif "Neural Network" in name:
        print(f"Best Hyperparameters: hidden_layer_sizes = {metrics['best_params']['hidden_layer_sizes']}, activation = {metrics['best_params']['activation']}")
    print("Confusion Matrix:")
    print(metrics['Confusion_matrix'])
    print("\nClassification Report:")
    print(metrics['report'])
print("========================================================")

#--- Comparing models --- #
print("=================== Comparing models ===================")
for name, metrics in result.items():
    print(f"{name}: Accuracy = {metrics['accuracy']*100:.2f}%")
print("========================================================")
final_model_name = max(result, key=lambda name: result[name]['accuracy'])
final_model = best_models[final_model_name]
print(f"Best performing model: {final_model_name}")
print("========================================================")

#--- Example 1 prediction ---#
#example1_passenger = pd.DataFrame([[3, 1, 25, 0, 0, 7.25, 0, 1, 0]], columns=X_train.columns) #3rd class, Male, 25 years old, no siblings/spouse, no parents/children, fare 7.25, embarked from Q
#example1_passenger_scaled = scaler.transform(example1_passenger)

#example2_passenger = pd.DataFrame([[1, 0, 28, 0, 0, 80, 1, 0, 0]], columns=X_train.columns) #1st class, Female, 28 years old, no siblings/spouse, no parents/children, fare 80, embarked from S
#example2_passenger_scaled = scaler.transform(example2_passenger)

Pclass = int(input("Enter Passenger Class (1, 2, or 3): "))

Gender=-1
while Gender not in [0, 1]:
    G = input("Is passenger male or female?").lower()

    if G == 'male' or G == 'm':
        Gender = 1
    elif G == 'female' or G =='f':
        Gender = 0
    else:
        print("Invalid input. Please enter 'male' or 'female'.")

Age = int(input("Enter Age of the passenger: "))
SibSp = int(input("Enter number of Siblings/Spouse aboard: "))
Parch = int(input("Enter number of Parents/Children aboard: "))
Fare = float(input("Enter Fare paid by the passenger: "))
Embarked = input("Where did the passenger embark from? (C, Q, S): ").upper()

if Embarked == 'C':
    Embarked_C, Embarked_Q, Embarked_S = 1, 0, 0
elif Embarked == 'Q':
    Embarked_C, Embarked_Q, Embarked_S = 0, 1, 0
elif Embarked == 'S':
    Embarked_C, Embarked_Q, Embarked_S = 0, 0, 1

example_passenger = pd.DataFrame([[Pclass, Gender, Age, SibSp, Parch, Fare, Embarked_C, Embarked_Q, Embarked_S]], columns=X_train.columns)
example_passenger_scaled = scaler.transform(example_passenger)

print("Example passenger 1 prediction (0=Not Survived, 1=Survived):", final_model.predict(example_passenger_scaled)[0])
