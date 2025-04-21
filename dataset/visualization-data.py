import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('NIFTY_50.csv')

print(df.head())
print(df.columns)

# Convert date column to datetime if needed
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])

# Select only numeric columns
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Compute correlation matrix
correlation_matrix = numeric_df.corr()

# Drop 'P/E' and 'P/B' from both rows and columns
corr_filtered = correlation_matrix.drop(columns=['P/E', 'P/B'], index=['P/E', 'P/B'])

# Plot the filtered heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_filtered, annot=True, cmap='coolwarm', vmin=0, vmax=1)
plt.title("Filtered Correlation Heatmap - NIFTY 50")

# Save to PNG
# plt.savefig("filtered_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# the plot shows strong positive correlation: when a value (e.g. open) increases, the others tend to increase as well

# The following line plot depicts positive correlation, as the line increases, and as one increases, the others
# increase as well.

# Plot Open, High, Low, Close over time
plt.figure(figsize=(14, 6))
plt.plot(df['Date'], df['Open'], label='Open')
plt.plot(df['Date'], df['High'], label='High')
plt.plot(df['Date'], df['Low'], label='Low')
plt.plot(df['Date'], df['Close'], label='Close')

plt.title('NIFTY 50 - Price Movement Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.savefig("line-positive-corr.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()
