import pandas as pd

# Load your data
df = pd.read_csv("tasa_interes.csv", parse_dates=["observation_date"], dayfirst=True)

# Convert 'observation_date' to datetime if it's not already
df['observation_date'] = pd.to_datetime(df['observation_date'])

# Calculate the quarterly average for DFF
df['quarter'] = df['observation_date'].dt.to_period('Q')
quarterly_avg = df.groupby('quarter')['DFF'].transform('mean')

# Add the calculated quarterly average to the original DataFrame
df['quarterly_avg'] = quarterly_avg

# Save the result to a new CSV file
df.to_csv("tasa_interes_with_quarterly_avg.csv", index=False)

# Display the result
print(df.head())
