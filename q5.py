import pandas as pd
import numpy as np

# Entropy function
def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))

# Information gain
def info_gain(data, split_attribute, target):
    total_entropy = entropy(data[target])
    values, counts = np.unique(data[split_attribute], return_counts=True)
    weighted_entropy = np.sum([
        (counts[i] / np.sum(counts)) * entropy(data[data[split_attribute] == values[i]][target])
        for i in range(len(values))
    ])
    return total_entropy - weighted_entropy

# Best attribute selection
def best_attribute(data, target):
    gains = {col: info_gain(data, col, target) for col in data.columns if col != target}
    return max(gains, key=gains.get)

# Recursive Decision Tree builder
def build_tree(data, target):
    # If all values are same -> leaf
    if len(np.unique(data[target])) == 1:
        return np.unique(data[target])[0]
    
    # If no features left -> majority class
    if len(data.columns) == 1:
        return data[target].mode()[0]

    # Choose best attribute
    best_feat = best_attribute(data, target)
    tree = {best_feat: {}}

    # Split on best attribute
    for value in np.unique(data[best_feat]):
        subset = data[data[best_feat] == value].drop(columns=[best_feat])
        tree[best_feat][value] = build_tree(subset, target)

    return tree

# Prediction using tree
def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree  # Leaf node
    root = next(iter(tree))
    value = sample[root]
    if value in tree[root]:
        return predict(tree[root][value], sample)
    else:
        return None  # unknown branch

# Load dataset
df = pd.read_csv(r"C:\Users\prana\OneDrive\Desktop\machine_learning\lab6\DCT_mal.csv")

# Assume last column is target
target = df.columns[-1]

# Build decision tree
tree = build_tree(df, target)
print("Decision Tree:", tree)

# Predict for first row
sample = df.iloc[0].to_dict()
print("Prediction for first row:", predict(tree, sample))
