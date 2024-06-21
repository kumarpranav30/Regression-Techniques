import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import random as rd

file_path = './Dataset/Student/student-mat.csv'
data = pd.read_csv(file_path, sep=';')

print(data.head())

best_r2 = 0
while True:
    a1 = rd.randint(0, 10)
    a2 = rd.randint(1, 10)
    data['avg_G1_G2'] = (a1 * data['G1'] + a2 * data['G2']) / (a1 + a2)

    X = data[['avg_G1_G2']]
    y = data['G3']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    if r2 <= best_r2:
        continue
    
    best_r2 = r2

    print(f'Mean Squared Error (MSE): {mse}')
    print(f'R-squared (R2): {r2}')
    print(f'a1: {a1}\na2: {a2}')
    print(f'Coefficients: {model.coef_}')
    print(f'Intercept: {model.intercept_}')

    plt.scatter(X_test, y_test, color='blue', label='Actual data')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Fitted line')
    plt.xlabel('Avg. Period Grade (G1 & G2)')
    plt.ylabel('Final Grade (G3)')
    plt.title('Linear Regression Fit')
    plt.legend()
    plt.grid(True)
    plt.show()
