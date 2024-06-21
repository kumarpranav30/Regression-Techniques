import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


file_path = './Dataset/Student/student-mat.csv'
data = pd.read_csv(file_path, sep=';')


data['internet'] = data['internet'].map({'yes': 1, 'no': 0})




selected_features = ['G1', 'G2', 'internet', 'failures', 'G3']


print(data[selected_features])


sorted_data = data[selected_features].sort_values(by='G3')



print("\nSorted by 'G3' column:")
print(sorted_data)

selected_features = ['G1', 'G2', 'internet', 'failures']
X = data[selected_features]
y = data['G3']  


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


model = LinearRegression()


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2): {r2}')


print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
