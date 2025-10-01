"""Electrochemical Surface Area (ECSA) Analysis and Visualization Script.

This script analyzes predicted ECSA values from multi-modal spectroscopic data,
categorizes samples by performance, and generates statistical visualizations
and mean spectroscopic signatures.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Spectroscopic measurement sequence
customsequence = ('EXAFS_K2', 'XRD_2_P', 'XANES', 'PDF', 'HAXPES_VB', 'SAXS', 'HAXPES_Pt3d', 'HAXPES_Pt4f')
DIR_Data = ""
filename = r""  # Path to Excel file with ECSA prediction results [converted from .txt]

# Load ECSA prediction results
df = pd.read_excel(filename, engine='openpyxl')

# Filter for valid ECSA predictions only
df = df[df['Predicted score'] > 0]

# Extract Electrochemical Surface Area predictions
predicted_ECSA = df['Predicted score']

# Extract spectroscopic feature columns
predicted_col_index = df.columns.get_loc('Predicted score')
augmented_data_columns = df.columns[predicted_col_index + 1:].tolist()

# Combine all spectroscopic features into a single list per sample
df['AugmentedData'] = df[augmented_data_columns].apply(lambda row: row.dropna().tolist(), axis=1)

# Define ECSA performance thresholds
low = 20  # Lower threshold for ECSA (m²/g-Pt)
high = 190 # Upper threshold for ECSA (m²/g-Pt)

# Categorize samples based on ECSA performance
df['Category'] = pd.cut(df['Predicted score'],
                        bins=[-float('inf'), low, high, float('inf')],
                        labels=['Bad Sample', 'Neutral Sample', 'Good Sample'],
                        right=False)

# Separate high and low performing ECSA samples
good_samples_df = df[df['Category'] == 'Good Sample']
bad_samples_df = df[df['Category'] == 'Bad Sample']

# Create dictionaries mapping ECSA values to spectroscopic features
good_samples_dict = good_samples_df.set_index('Predicted score')['AugmentedData'].to_dict()
bad_samples_dict = bad_samples_df.set_index('Predicted score')['AugmentedData'].to_dict()

mean_good_keys = np.mean(list(good_samples_dict.keys()))
mean_bad_keys = np.mean(list(bad_samples_dict.keys()))

# Report sample distribution statistics
print("good count", len(good_samples_dict), "low", low, "high", high)
print("bad count", len(bad_samples_dict), "low", low, "high", high)

# Generate ECSA distribution plot with performance thresholds
plt.figure(figsize=(10, 6))
sns.histplot(predicted_ECSA, kde=True, label='Predicted ECSA')
plt.axvline(mean_good_keys, color='r', linestyle='--', label=f'Mean Good Sample Score: {mean_good_keys:.2f}')
plt.axvline(mean_bad_keys, color='b', linestyle='--', label=f'Mean Bad Sample Score: {mean_bad_keys:.2f}')
plt.xlabel('Predicted ECSA')
plt.ylabel('Frequency')
plt.title(f'Distribution of Predicted ECSA')
plt.legend()
plt.savefig(f"{DIR_Data}/distribution_Predicted_ECSA.png")
plt.show()

def calculate_elementwise_mean(data_dict):
    # Convert lists to numpy arrays for element-wise operations
    data_arrays = [np.array(data_list) for data_list in data_dict.values()]
    # Stack arrays vertically and compute the mean along axis 0
    stacked_data = np.stack(data_arrays)
    mean_data = np.mean(stacked_data, axis=0)
    return mean_data

good_samples_mean = []

if len(good_samples_dict) != 0:
    good_samples_mean = calculate_elementwise_mean(good_samples_dict)
    a = good_samples_mean[:300]

bad_samples_mean = []

if len(bad_samples_dict) != 0:
    bad_samples_mean = calculate_elementwise_mean(bad_samples_dict)
    b = bad_samples_mean[:300]

# Create a DataFrame for each sample
df_good = pd.DataFrame({'Good Samples Mean': good_samples_mean})
df_bad = pd.DataFrame({'Bad Samples Mean': bad_samples_mean})

# Save to Excel
df_good.to_excel(f'{DIR_Data}good_samples_mean_II_ECSA.xlsx', index=False)
df_bad.to_excel(f'{DIR_Data}bad_samples_mean_II_ECSA.xlsx', index=False)

# Plot the mean augmented data for Good and Bad samples
plt.figure(figsize=(25, 6))
plt.plot(good_samples_mean, label='Good Samples')
plt.plot(bad_samples_mean, label='Bad Samples')
plt.title(f'Mean Augmented Data for Samples ECSA')
plt.legend()
plt.savefig(f"{DIR_Data}/Mean_Sample_ECSA.png")
plt.show()

# Save the mean augmented data to an Excel file
if len(bad_samples_mean) > 0 and len(good_samples_mean) > 0:
    output_df = pd.DataFrame({
        'Good Data': good_samples_mean,
        'Bad Data': bad_samples_mean
    })
    output_file_path = f"{DIR_Data}/good_bad_samples_mean_data_ECSA.xlsx"
    output_df.to_excel(output_file_path, header=False)
elif len(bad_samples_mean) > 0:
    output_df = pd.DataFrame({
        'Bad Data': bad_samples_mean
    })
    output_file_path = f"{DIR_Data}/good_bad_samples_mean_data_ECSA.xlsx"
    output_df.to_excel(output_file_path, header=False)
elif len(good_samples_mean) > 0:
    output_df = pd.DataFrame({
        'Good Data': good_samples_mean
    })
    output_file_path = f"{DIR_Data}/good_bad_samples_mean_data_ECSA.xlsx"
    output_df.to_excel(output_file_path, header=False)
else:
    print("Nothing found")
