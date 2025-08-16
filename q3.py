import pandas as pd
import math
#The function find_root_node_with_info_gain bins numeric features into categories, calculates the Information Gain for each feature using entropy, and returns the feature with the highest gain as the decision tree’s root node.
def find_root_node_with_info_gain(data, target_column, bins=3):
    #  Bin numeric features into categorical bins
    for column in data.columns:
        if column != target_column:
            data[column] = pd.cut(data[column], bins=bins, labels=False)

    #  Calculate dataset entropy
    label_counts = data[target_column].value_counts(normalize=True)
    total_entropy = 0
    for p in label_counts:
        if p > 0:
            total_entropy += -p * math.log2(p)

    #  Calculating Information Gain 
    ig_scores = {}
    for feature in data.columns:
        if feature == target_column:
            continue
        feature_values = data[feature].unique()
        weighted_entropy = 0
        for value in feature_values:
            subset = data[data[feature] == value]
            subset_size = len(subset) / len(data)
            subset_label_counts = subset[target_column].value_counts(normalize=True)
            subset_entropy = 0
            for p in subset_label_counts:
                if p > 0:
                    subset_entropy += -p * math.log2(p)
            weighted_entropy += subset_size * subset_entropy
        ig = total_entropy - weighted_entropy
        ig_scores[feature] = ig

    #  Finding feature with maximum IG
    best_feature = max(ig_scores, key=ig_scores.get)
    return best_feature, ig_scores

file_path = r"C:\Users\prana\OneDrive\Desktop\machine_learning\lab6\DCT_mal.csv"
dataset = pd.read_csv(file_path)
best_feature, ig_scores = find_root_node_with_info_gain(dataset, target_column='LABEL', bins=3)
print("Information Gain Scores:")
for feature, score in ig_scores.items():
    print(f"{feature}: {score:.4f}")
print(f"\nRoot Node Feature: {best_feature}")

