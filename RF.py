import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


file_path = './Dataset/Student/student-mat.csv'
data = pd.read_csv(file_path, sep=';')


data['address'] = data['address'].map({'U': 1, 'R': 0})


selected_features = ['G1', 'G2', 'studytime', 'famrel', 'address']
X = data[selected_features]
y = data['G3']  


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


rf_model = RandomForestRegressor(random_state=42, n_estimators=100)


rf_model.fit(X_train, y_train)


y_pred = rf_model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2): {r2}')


plt.figure(figsize=(20,10))
tree = rf_model.estimators_[0]
plot_tree(tree, feature_names=selected_features, filled=True, rounded=True, fontsize=10)
plt.show()

import numpy as np


importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [selected_features[i] for i in indices], rotation=45)
plt.xlim([-1, X.shape[1]])
plt.show()
