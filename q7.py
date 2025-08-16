import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

def load_data(file_path, target_column, feature1, feature2):
    # Load dataset and return selected features and target
    df = pd.read_csv(file_path)
    X = df[[feature1, feature2]]
    y = df[target_column]
    return X, y

def train_decision_tree(X, y, max_depth=4):
    # Train a Decision Tree classifier on the given features
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth, random_state=42)
    clf.fit(X, y)
    return clf

def plot_decision_boundary(clf, X, y, feature1, feature2):
    # Plot the decision boundary created by the trained Decision Tree
    x_min, x_max = X[feature1].min() - 1, X[feature1].max() + 1
    y_min, y_max = X[feature2].min() - 1, X[feature2].max() + 1

    # Limit resolution to avoid memory issues (max 500x500 grid)
    grid_points = 500
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_points),
        np.linspace(y_min, y_max, grid_points)
    )

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    scatter = plt.scatter(X[feature1], X[feature2], c=y, edgecolor="k", cmap=plt.cm.RdYlBu)
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title("Decision Tree Decision Boundary")
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.show()

# ---------------- MAIN CODE ---------------- #

file_path = r"C:\Users\prana\OneDrive\Desktop\machine_learning\lab6\DCT_mal.csv"
target_column = "LABEL"    # replace with actual target column
feature1 = "1"       # replace with actual feature column 1
feature2 = "2"       # replace with actual feature column 2

# Load dataset
X, y = load_data(file_path, target_column, feature1, feature2)
print("Data loaded with features:", feature1, "and", feature2)

# Train Decision Tree
clf = train_decision_tree(X, y)
print("Decision Tree trained successfully.")

# Visualize decision boundary
plot_decision_boundary(clf, X, y, feature1, feature2)
