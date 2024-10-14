import pandas as pd  # For data handling
import numpy as np   # For numerical operations
import seaborn as sns  # For visualizing data
import matplotlib.pyplot as plt  # For plotting graphs
from sklearn.model_selection import train_test_split  # For splitting the dataset
from sklearn.linear_model import LogisticRegression  # For Logistic Regression model
from sklearn.ensemble import RandomForestClassifier  # For Random Forest model
from sklearn.metrics import accuracy_score, classification_report  # For model evaluation


# Load the Titanic dataset
df = pd.read_csv('Titanic-Dataset.csv')

# Display the first few rows of the dataset
df.head()


# Check for missing data
df.isnull().sum()

# Fill missing 'Age' values with the median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing 'Embarked' values with the most frequent value (mode)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop the 'Cabin' column because it has too many missing values
df.drop('Cabin', axis=1, inplace=True)



# Convert 'Sex' to numerical values: male = 0, female = 1
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# One-hot encode 'Embarked' (creates separate columns for each category)
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)


# Drop irrelevant columns
df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)


# Create a new feature 'FamilySize' by adding 'SibSp' and 'Parch'
df['FamilySize'] = df['SibSp'] + df['Parch']

# Create a new feature 'IsAlone' based on 'FamilySize'
df['IsAlone'] = 0
df['IsAlone'].loc[df['FamilySize'] == 0] = 1  # If no family, IsAlone = 1


# Define the feature matrix (X) and target variable (y)
X = df.drop('Survived', axis=1)  # Features
y = df['Survived']  # Target variable

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)  # Train the model

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))



# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)  # Train the model

# Make predictions on the test set
rf_pred = rf_model.predict(X_test)

# Evaluate the Random Forest model
print(f'Random Forest Accuracy: {accuracy_score(y_test, rf_pred)}')



# Import necessary metric for ROC curve
from sklearn.metrics import roc_curve, roc_auc_score

# Get predicted probabilities
y_proba = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

# Plot ROC curve
plt.plot(fpr, tpr, label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# Calculate the Area Under the Curve (AUC) score
auc_score = roc_auc_score(y_test, y_proba)
print(f'AUC Score: {auc_score}')



