import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

# 1. Load dataset
df = pd.read_csv(r"C:\Users\prana\OneDrive\Desktop\machine_learning\lab6\DCT_mal.csv")

# Check first few rows
print("Data Preview:")
print(df.head())
print("Columns:", df.columns)

# 2. Separate features (X) and target (y)
#  Replace 'target_column' with your actual target column name
target_column = 'LABEL'   # <-- update this
y = df[target_column]
X = df.drop(columns=[target_column])

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Decision Tree using Entropy
clf = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# 5. Visualize the Decision Tree
plt.figure(figsize=(20,10))
plot_tree(clf,
          feature_names=X.columns,
          class_names=[str(c) for c in clf.classes_],
          filled=True, rounded=True, fontsize=10)
plt.show()
