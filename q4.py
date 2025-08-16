import pandas as pd
import numpy as np

# entropy calculation
def entropy(y):
    probs = y.value_counts(normalize=True)  # faster than loop
    return -np.sum(probs * np.log2(probs))

# information gain calculation
def information_gain(df, feature, target):
    total_entropy = entropy(df[target])
    values = df[feature].value_counts(normalize=True)
    weighted_entropy = sum(values[v] * entropy(df[df[feature] == v][target]) for v in values.index)
    return total_entropy - weighted_entropy

# load dataset
df = pd.read_csv(r"C:\Users\prana\OneDrive\Desktop\machine_learning\lab6\DCT_mal.csv")

target = df.columns[-1]  # last column as target
features = df.columns[:-1]

# compute IG for all features
ig_scores = {feature: information_gain(df, feature, target) for feature in features}

# find best root
root = max(ig_scores, key=ig_scores.get)

print("Information Gain for each feature:")
for k, v in ig_scores.items():
    print(f"{k}: {v:.4f}")

print("\nRoot Node Feature:", root)
