import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


file_path = "./Dataset/Student/student-mat.csv"
data = pd.read_csv(file_path, sep=";")

data["address"] = data["address"].map({"U": 1, "R": 0})


selected_features = ["G1", "G2", "studytime", "famrel", "address", "G3"]

data = data[selected_features]

print(data.head())


X = data.drop(columns=['G3'])  
y = data['G3']  


pearson_correlations = {}
for feature in X.columns:
    corr, _ = pearsonr(X[feature], y)
    pearson_correlations[feature] = corr


spearman_correlations = {}
for feature in X.columns:
    corr, _ = spearmanr(X[feature], y)
    spearman_correlations[feature] = corr


pearson_sorted = sorted(pearson_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
spearman_sorted = sorted(spearman_correlations.items(), key=lambda x: abs(x[1]), reverse=True)


features = [feat for feat, _ in pearson_sorted]
pearson_values = [corr for _, corr in pearson_sorted]
spearman_values = [corr for _, corr in spearman_sorted]


plt.figure(figsize=(12, 6))
bar_width = 0.35
index = range(len(features))

plt.bar(index, pearson_values, bar_width, label='Pearson')
plt.bar([i + bar_width for i in index], spearman_values, bar_width, label='Spearman')

plt.xlabel('Features')
plt.ylabel('Correlation Coefficient')
plt.title('Feature Correlation with G3')
plt.xticks([i + bar_width / 2 for i in index], features, rotation=45, ha='right')
plt.legend()

plt.tight_layout()
plt.show()
