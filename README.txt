Project: Titanic Survival Prediction - Machine Learning Project
Course: CSCI 5371 - Machine Learning
Name: Shivam Patel


-------------------------------------
Description
-------------------------------------
This project predicts whether a passenger survived the Titanic disaster using multiple machine learning models.

The workflow includes data cleaning, feature engineering, preprocessing, model training, model comparison, and an interactive prediction
section where user can input new passenger details. 


-------------------------------------
Project Overview
-------------------------------------
The goal of this project is to built and compare several ML classifiers to determine which performs best in predicting Titanic passenger survival.

The following models were trained and evaluated
	- Logistic regression (With multiple C values)
	- K-Nearest Neighbors (different distance & neighbors)
	- Neural Network (MLPClassifier) (Various architecture)

The model with the highest best accuracy is automatically selected as the final best model.

-------------------------------------
Dataset
-------------------------------------
The dataset is loaded directly from a public Github Source:
	https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

It includes the following key features:
- Passenger Class (Pclass)
- Gender (Sex)
- Age
- Siblings/spouse count (SibSp)
- Parents/children count (parch)
- Fare
- Embarked location

The target variable is:
- Survived (0 = Did not survived, 1 = Survived)

-------------------------------------
Data Cleaning & Preprocessing
-------------------------------------
The script performs multiple cleaning steps:
01. Handle missing values
	- Fill missing Age with mean
	- Fill missing Embarked with the mode

02. Drop irrelevant features
	- Removed columns include Cabin, Ticket, Name

03. Encode categorical data
	- Convert sex -> (male=1, female=0)
	- One-hot encode Embarked -> (Embarked_C, Embarked_Q, Embarked_S)

04. Feature scaling
	- StandardScaler is used so models especially kNN & Neural Network perform correctly

-------------------------------------
Models Trained
-------------------------------------
The project trains multiple variations of each model:
01. Logistic regression
	- C = 0.1
	- C = 1.0
	- C = 10.0

02. kNN Classifiers
	- K = 3 (Euclidean)
	- K = 5 (Distance-weighted)
	- K = 7 (Manhattan distance)

03. Neural Networks (MLP)
	- 1 hidden layer: (8,)
	- 2 hidden layers: (16, 8)
	- 3 hidden layers: (32, 16, 8)

Each model is evaluated using:
	- Accuracy score
	- Classification report
	- Confusion matrix
The best model for each family is stored and compared

-------------------------------------
Model Selection
-------------------------------------
The script automatically selects the best-performing model based on accuracy from the test set.

relevant script:
final_model_name = max(result, key=lambda name: result[name]['accuracy'])
final_model = best_models[final_model_name]

The chosen model is then used for user predictions.

-------------------------------------
User prediction
-------------------------------------
At the end of the script, the user can enter real values:
- Passenger class (1/2/3)
- Gender (male/female)
- Age
- Sibsp
- Parch
- Fare
- Embarked location (C/Q/S)

This script converts the input into models required format, scales them, and prints:
0 -> Not survived
1 -> Survived

This allows real-time survival prediction

-------------------------------------
How to run the script
-------------------------------------
Install dependencies if not installed:
	pip install pandas numpy scikit-learn

Run:
	python3 Project.py

-------------------------------------
Outputs
-------------------------------------
The script prints:
- Model accuracy comparison
- Best model selection
- Confusion matrices
- Classification reports
- User prediction result

-------------------------------------
Summary
-------------------------------------
This project demonstrate a complete end-to-end machine learning workflow:
- Data acquisition
- Cleaning & preprocessing
- Feature engineering
- Model training
- Hyperparameter tuning
- Performance evaluation
- Real-time prediction

It highlights skills in Python, scikit-learn, data analysis, and ML model comparison.
