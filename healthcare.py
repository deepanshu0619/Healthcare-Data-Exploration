import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Healthcare Data from CSV file
df = pd.read_csv("healthcare_data.csv")

# Display basic info and summary
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Handle missing values (Simple imputation with mean for numeric columns)
df.fillna(df.mean(), inplace=True)

# Visualize distributions
plt.figure(figsize=(10, 6))
sns.pairplot(df)
plt.show()

# Visualize correlations
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Histogram for each numerical feature
df.hist(figsize=(12, 8), bins=20)
plt.suptitle("Feature Distributions")
plt.show()

# Boxplot for outlier detection
plt.figure(figsize=(12, 6))
sns.boxplot(data=df)
plt.xticks(rotation=45)
plt.title("Boxplot for Outlier Detection")
plt.show()

# Select features and target variable (Assuming 'Outcome' is the target column)
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train AI model (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))
