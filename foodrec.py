import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset (replace 'your_data.csv' with your CSV file path)
data = pd.read_csv('C:\\Users\\saach\\Downloads\\merged\\merged_data.csv')

# Display the first few rows of the dataset to understand its structure
print(data.head())

# Assuming 'mood', 'cuisine_type', 'health_level' are features, and 'meal_recommendation' is the target
features = ['breakfast', 'moodRange', 'health_level']
target = 'meal_recommendation'

# Encoding categorical variables
encoder = LabelEncoder()
data[features] = data[features].apply(encoder.fit_transform)

# Split the data into features (X) and target (y)
X = data[features]
y = data[target]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict the target values using the test set
y_pred = model.predict(X_test)

# Evaluate the model performance
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Example of meal recommendation based on mood, cuisine type, and health level
sample_data = pd.DataFrame({
    'mood': [encoder.transform(['happy'])[0]], # Replace with the desired mood
    'cuisine_type': [encoder.transform(['italian'])[0]], # Replace with the desired cuisine
    'health_level': [encoder.transform(['high'])[0]] # Replace with the desired health level
})

recommended_meal = model.predict(sample_data)
recommended_meal = encoder.inverse_transform(recommended_meal)

print(f'Recommended meal: {recommended_meal[0]}')
