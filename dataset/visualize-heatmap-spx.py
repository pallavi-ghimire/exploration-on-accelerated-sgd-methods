import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("SPX_clean.csv")

# Select the features of interest
features = ['MA_20', 'STD_20', 'MA_10', 'Bollinger_Width', 'Lagged_Return_1']
data = df[features]

# Compute the correlation matrix
corr_matrix = data.corr()

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Engineered Features")
plt.tight_layout()

# Save as SVG
plt.savefig("../results/heatmap_features.svg", format="svg")
plt.close()
