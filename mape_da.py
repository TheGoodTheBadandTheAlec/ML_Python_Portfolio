import pandas as pd
from sklearn.metrics import mean_squared_error
import ast

# Load the data from the CSV file
data = pd.read_csv(r'C:\Users\alecj\python\Crypto\ml_data_analysis.csv')

# Convert 'close' and 'predicted_close' columns to numeric data types and handle non-numeric values
data['close'] = pd.to_numeric(data['close'], errors='coerce')
data['predicted_close'] = data['predicted_close'].apply(lambda x: ast.literal_eval(x)[0] if pd.notna(x) else x)
data['predicted_close'] = pd.to_numeric(data['predicted_close'], errors='coerce')

# Drop rows with NaN values in 'close' or 'predicted_close'
data = data.dropna(subset=['close', 'predicted_close'])

# Create a DataFrame to store the results
results = []

# Calculate metrics for each symbol
symbols = data['symbol'].unique()

for symbol in symbols:
    symbol_data = data[data['symbol'] == symbol]
    actual = symbol_data['close']
    predicted = symbol_data['predicted_close']

    # Calculate MAPE
    mape = (abs(actual - predicted) / actual).mean()
    
    # Calculate DA
    direction_accuracy = (actual.shift(-1) - actual) * (predicted.shift(-1) - predicted) > 0
    da = direction_accuracy.mean()

    # Additional metrics
    mse = mean_squared_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted, squared=False)
    r2 = 1 - (mse / actual.var())
    
    # Append results to the list
    results.append({
        'Symbol': symbol,
        'MAPE': mape,
        'DA': da,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    })

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Save the results to a CSV file
output_csv_path = r'C:\Users\alecj\python\Crypto\evaluation_results.csv'
results_df.to_csv(output_csv_path, index=False)
print(f'Results saved to {output_csv_path}')
