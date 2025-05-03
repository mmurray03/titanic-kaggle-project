# -*- coding: utf-8 -*-
"""
@author: soeren and michael
"""

import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score

############################################## FUNCTIONS ######################################

"""

Adjusts Titanic passenger fares to individual ticket cost based on ticket number group size.
This gives a more accurate picture of what fare each passenger actually paid.

"""
def adjust_fares(data, data2, isTestData):

    # Create a copy of the input data
    data = data.copy()
    
    # Calculate group size based on ticket number
    ticket_counts = data['Ticket'].value_counts()
    data['GroupSize'] = data['Ticket'].map(ticket_counts)
    
    
    # Use GroupSize for fare adjustment by default
    data['ActualGroupSize'] = data['GroupSize']
    
    # Fallback to FamilySize when a row is missing fare or has zero fare
    data['ActualGroupSize'] = np.where(
        (data['Fare'].isna()) | (data['Fare'] == 0),
        data['FamilySize'], 
        data['ActualGroupSize']       )
    
    
    # Calculate adjusted fare (original fare / group size)
    data['AdjustedFare'] = data['Fare'] / data['ActualGroupSize']
    
    if (isTestData):
        
        # Calculate mean adjusted fare per class
        class_mean_fares = data2.groupby('Pclass')['AdjustedFare'].mean()
        
    else:
        
        class_mean_fares = data.groupby('Pclass')['AdjustedFare'].mean()
        
    
    # Replace 0 or NaN adjusted fares with the mean adjusted fare for the passenger's class
    for pclass in class_mean_fares.index:
        data.loc[(data['Pclass'] == pclass) & 
               ((data['AdjustedFare'] == 0) | (data['AdjustedFare'].isna())), 
               'AdjustedFare'] = class_mean_fares[pclass]
    
    
    # Round results
    data['AdjustedFare'] = data['AdjustedFare'].round(2)
    
    return data



########################################## DATA WRANGLING #########################################

# Import data
data = pd.read_csv('train.csv')

test = pd.read_csv('test.csv')


#------------------ TRAIN DATA ------------------------------
# Title extraction feature
data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Group rare titles into 'Rare' and standardize others
data['Title'] = data['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
data['Title'] = data['Title'].replace('Mlle', 'Miss').replace('Ms', 'Miss').replace('Mme', 'Mrs')


# Create Family Size feature
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

# Create Is Alone feature
data['IsAlone'] = (data['FamilySize'] == 1).astype(int)

# Create Deck feature
data['Deck'] = data['Cabin'].str[0].fillna('Unknown')


# Adjust Fares to reflect price for a single ticket for groups
data = adjust_fares( data, data, False )
# failed - data['Fare'] = data['Fare'].fillna(data['Fare'].median())

# Handle missing values in Embarked, Title, and Age
data['Embarked'].fillna('S', inplace=True)
data['Title'].fillna('Unknown', inplace=True)

# Impute Age
age_by_title = data.groupby('Title')['Age'].median()
data['Age'] = data.apply( lambda x: age_by_title[ x['Title'] ] if pd.isna( x['Age'] ) else x['Age'], axis=1 )


# Create bins for Age and Adjusted Fare
data['AgeBin'] = pd.qcut(data['Age'], q=5, labels=False)
data['FareBin'] = pd.qcut(data['AdjustedFare'], q=5, labels=False)
# failed - data['FareBin'] = pd.qcut(data['Fare'], q=5, labels=False)


#------------------ TEST DATA ------------------------------
# Title extraction feature
test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Group rare titles into 'Rare' and standardize others
test['Title'] = test['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
test['Title'] = test['Title'].replace('Mlle', 'Miss').replace('Ms', 'Miss').replace('Mme', 'Mrs')


# Create Family Size feature
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1

# Create Is Alone feature
test['IsAlone'] = (test['FamilySize'] == 1).astype(int)

# Create Deck feature
test['Deck'] = test['Cabin'].str[0].fillna('Unknown')


# Adjust Fares to reflect price for a single ticket for groups
test = adjust_fares( test, data, True )
# failed - test['Fare'] = test['Fare'].fillna(data['Fare'].median())


# Handle missing values in Embarked, Title, and Age
test['Embarked'].fillna('S', inplace=True)
test['Title'].fillna('Unknown', inplace=True)

# Impute Age
test['Age'] = test.apply( lambda x: age_by_title[ x['Title'] ] if pd.isna( x['Age'] ) else x['Age'], axis=1 )


# Create bins for Age and Adjusted Fare
test['AgeBin'] = pd.qcut(test['Age'], q=5, labels=False)
test['FareBin'] = pd.qcut(test['AdjustedFare'], q=5, labels=False)
# failed - test['FareBin'] = pd.qcut(test['Fare'], q=5, labels=False)




#--------------------------- FEATURE ENCODING -----------------------------
# Encode features

X = data[['Pclass', 'Sex', 'Age', 'AdjustedFare', 'Embarked', 'Title', 'FamilySize', 'IsAlone', 'AgeBin', 'FareBin', 'Deck']]
# failed - X = data[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone', 'AgeBin', 'FareBin', 'Deck']]
#X = data[['Sex', 'Age', 'AdjustedFare', 'Title', 'FamilySize']] - failed

y = data['Survived']

X_test = test[X.columns]



# Target encode Embarked, Title, and Deck - Keep AgeBin and FareBin ordinal 

encoder = TargetEncoder(cols=['Embarked', 'Title', 'Deck'], smoothing=1)
# failed - encoder = TargetEncoder(cols=['Title'], smoothing=1) - failed

X = encoder.fit_transform(X, y)

X_test = encoder.transform(X_test)


# # Try one-hot encoding instead - Failed
# X = pd.get_dummies(X, columns=['Embarked', 'Title', 'Deck'], prefix=['Emb', 'Title', 'Deck'])
# X_test = pd.get_dummies(X_test, columns=['Embarked', 'Title', 'Deck'], prefix=['Emb', 'Title', 'Deck'])
# # Align train and test columns
# X, X_test = X.align(X_test, join='left', axis=1, fill_value=0) - failed



# Encode Sex

X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})

X_test['Sex'] = X_test['Sex'].map({'male': 0, 'female': 1})



# Scale numerical features

scaler = StandardScaler()

num_cols = ['Age', 'AdjustedFare', 'FamilySize']
# failed - num_cols = ['Age', 'Fare', 'FamilySize']

X[num_cols] = scaler.fit_transform(X[num_cols])

X_test[num_cols] = scaler.transform(X_test[num_cols])



################################# FEATURE IMPORTANCE ##############################
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
print(importance.sort_values(by='Importance', ascending=False))



################################ HYPERPARAMETER TUNING ##############################
"""
# Base models
base_models = [
    ('rf', RandomForestClassifier(class_weight='balanced', random_state=42)),
    ('xgb', XGBClassifier(random_state=42)),
    ('lr', LogisticRegression(class_weight='balanced', random_state=42)),
    ('cat', CatBoostClassifier(verbose=0, random_state=42)),
    ('svm', SVC(probability=True, random_state=42))
]

# Meta-model
meta_model = LogisticRegression()

stack = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)

param_grid = {
    'rf__n_estimators': [50, 100, 200],
    'rf__max_depth': [3, 5, 7],
    
    'xgb__n_estimators': [50, 100, 200],
    'xgb__max_depth': [3, 4, 5],
    'xgb__learning_rate': [0.01, 0.1, 0.3],
    
    'lr__C': [0.01, 0.1, 1.0],
    
    'cat__iterations': [100, 200],
    'cat__depth': [4, 6],
    'cat__learning_rate': [0.01, 0.1],
    
    'svm__C': [0.1, 1.0, 10.0],
    'svm__kernel': ['rbf']
}

# Scoring metric
scorer = make_scorer(f1_score, average='weighted')


"""
#Tune hyperparameters for stacking classifier.

"""
def tune_stacking_classifier(X, y, n_iter=25):
    
    random_search = RandomizedSearchCV(
        estimator=stack,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=5,
        scoring=scorer,
        n_jobs=-1,
        verbose=1,
        random_state=42,
        return_train_score=True
    )
    
    random_search.fit(X, y)
    
    # Best model and parameters
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    
    return best_model, best_params, best_score


best_model, best_params, best_score = tune_stacking_classifier(X, y, n_iter = 50)

# Print results
print("Best Parameters:", best_params)
print("Best Cross-validation Score:", best_score)
"""


################################# MODEL SELECTION ################################
# # Define stacking ensemble
# base_models = [
#     ('rf', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
#     ('xgb', XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42))
# ]
# meta_model = LogisticRegression()
# stack = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)

#stack = best_model

base_models = [
    ('rf', RandomForestClassifier(class_weight='balanced', n_estimators=200, max_depth=5, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)),
    ('lr', LogisticRegression(class_weight='balanced', C=1.0, random_state=42)),
    ('cat', CatBoostClassifier(verbose=0, learning_rate=0.01, iterations=100, depth=4, random_state=42)),
    ('svm', SVC(kernel='rbf', C=1.0, probability=True, random_state=42))
]

# Meta-model
meta_model = LogisticRegression()

stack = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=10)


# Evaluate with cross-validation
scores = cross_val_score(stack, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='accuracy')
print(f"Stacking CV Accuracy: {scores.mean():.4f}")

# Fit model
stack.fit(X, y)

# Submission
predictions = stack.predict(X_test)

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})                                                                        
output.to_csv('submission.csv', index=False)
print("\n\nYour submission was successfully saved!")









