import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the dataset
file_path = './Dataset/Student/student-mat.csv'
data = pd.read_csv(file_path, sep=';')

# Convert 'internet' column to integer (yes: 1, no: 0)
data['address'] = data['address'].map({'U': 1, 'R': 0})

# Selecting features and target variable
selected_features = ['G1', 'G2', 'studytime', 'famrel', 'address']
X = data[selected_features]
y = data['G3']  # Target variable

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the parameter grid for 'degree'
param_grid = {'degree': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

# Initialize SVR model with polynomial kernel
svr = SVR(kernel='poly')

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1)

# Fit the GridSearchCV to find the best 'degree'
grid_search.fit(X_train, y_train)

# Get the best parameters
best_degree = grid_search.best_params_['degree']
print(f'Best degree for polynomial kernel: {best_degree}')

# Train the SVR model with the best degree
best_svr = SVR(kernel='poly', degree=best_degree)
best_svr.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_svr.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2): {r2}')
