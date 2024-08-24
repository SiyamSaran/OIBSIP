import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv(r"E:\Advertising.csv")

# Display the first few rows of the dataset
print(data.head(10))

# Check for missing values
print(data.isnull().sum())

# Drop or fill missing values
data = data.dropna()

data = pd.get_dummies(data, drop_first=True)

# Define features and target variable
X = data.drop('Sales', axis=1)  # Features
y = data['Sales']               # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Line for perfect prediction
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.show()

import joblib
joblib.dump(model, 'sales_prediction_model.pkl')
print(model.feature_names_in_)



# User inputs
tv_spend = float(input("Enter the TV advertising spend: "))
radio_spend = float(input("Enter the Radio advertising spend: "))
newspaper_spend = float(input("Enter the Newspaper advertising spend: "))

# Create a DataFrame with the input values
new_data = pd.DataFrame({
    'TV': [tv_spend],           
    'Radio': [radio_spend],    
    'Newspaper': [newspaper_spend]  
})

new_prediction = model.predict(new_data)
print(f"Predicted Sales: {new_prediction[0]}")
