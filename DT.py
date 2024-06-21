import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the dataset
file_path = './Dataset/Student/student-mat.csv'
data = pd.read_csv(file_path, sep=';')

# Convert 'address' column to integer (U: 1, R: 0)
data['address'] = data['address'].map({'U': 1, 'R': 0})

# Selecting features and target variable
selected_features = ['G1','G2']
X = data[selected_features]
y = data['G3']  # Target variable

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the DecisionTreeRegressor model
tree_model = DecisionTreeRegressor(random_state=42)

# Train the model
tree_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = tree_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2): {r2}')

# Optionally, you can visualize the decision tree (requires graphviz and matplotlib)
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(200,100))
plot_tree(tree_model, feature_names=selected_features, filled=True, rounded=True, fontsize=10)
plt.show()
