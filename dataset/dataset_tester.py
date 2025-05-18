import pandas as pd

spx = pd.read_csv('SPX.csv')

# Ensure datetime and sort
spx['Date'] = pd.to_datetime(spx['Date'])
spx = spx.sort_values('Date').reset_index(drop=True)

# Use 'Close' price to compute features and target
spx['Return'] = spx['Close'].pct_change()
spx['MA_20'] = spx['Close'].rolling(window=20).mean()
spx['STD_20'] = spx['Close'].rolling(window=20).std()
spx['Z_Score'] = (spx['Close'] - spx['MA_20']) / spx['STD_20']

# Additional example features
spx['MA_10'] = spx['Close'].rolling(window=10).mean()
spx['Bollinger_Width'] = (spx['MA_20'] + 2*spx['STD_20']) - (spx['MA_20'] - 2*spx['STD_20'])
spx['Lagged_Return_1'] = spx['Return'].shift(1)

# Drop rows with NaN values from rolling windows
spx_clean = spx.dropna().copy()

# Prepare features and target
features = ['MA_10', 'MA_20', 'STD_20', 'Bollinger_Width', 'Lagged_Return_1']
target = 'Z_Score'  # this means R^1

print(spx_clean.head())

spx_clean.to_csv('SPX_clean.csv', index=False)



