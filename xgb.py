import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np

file_path = './Dataset/Student/student-mat.csv'
data = pd.read_csv(file_path, sep=';')

data['address'] = data['address'].map({'U': 1, 'R': 0})

selected_features = ['G1', 'G2', 'studytime', 'famrel', 'address']
X = data[selected_features]
y = data['G3']  # Target variable

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

xg_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)

xg_reg.fit(X_train, y_train)

y_pred = xg_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2): {r2}')

importances = xg_reg.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), [selected_features[i] for i in indices])
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()
