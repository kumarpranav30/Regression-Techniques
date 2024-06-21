import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
file_path = './Dataset/Student/student-mat.csv'
data = pd.read_csv(file_path, sep=';')

# Convert 'higher' column to integer (yes: 1, no: 0)
data['internet'] = data['internet'].map({'yes': 1, 'no': 0})
# data['address'] = data['address'].map({'U': 1, 'R': 0})
# data['Pstatus'] = data['Pstatus'].map({'T': 1, 'A': 0})

# Selecting features and target variable
selected_features = ['G1', 'G2', 'internet', 'failures', 'G3']

# Print the selected features
print(data[selected_features])

# Sort data[selected_features] by 'G3' column
sorted_data = data[selected_features].sort_values(by='G3')

# pd.set_option('display.max_rows', None)

print("\nSorted by 'G3' column:")
print(sorted_data)
# exit()
selected_features = ['G1', 'G2', 'internet', 'failures']
X = data[selected_features]
y = data['G3']  # Target variable

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2): {r2}')

# Optionally, you can print the coefficients and intercept
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
