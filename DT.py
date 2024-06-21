import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


file_path = './Dataset/Student/student-mat.csv'
data = pd.read_csv(file_path, sep=';')


data['address'] = data['address'].map({'U': 1, 'R': 0})


selected_features = ['G1','G2']
X = data[selected_features]
y = data['G3']  


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


tree_model = DecisionTreeRegressor(random_state=42)


tree_model.fit(X_train, y_train)


y_pred = tree_model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2): {r2}')


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(200,100))
plot_tree(tree_model, feature_names=selected_features, filled=True, rounded=True, fontsize=10)
plt.show()
