"""Specific Activity (SA) Analysis and Visualization Script.

This script analyzes predicted Specific Activity values from multi-modal
spectroscopic data, categorizes samples by performance, and generates
statistical visualizations and mean spectroscopic signatures.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Spectroscopic measurement sequence
customsequence = ('EXAFS_K2', 'XRD_2_P', 'XANES', 'PDF', 'HAXPES_VB', 'SAXS', 'HAXPES_Pt3d', 'HAXPES_Pt4f')
DIR_Data = ""
filename = r""  # Path to Excel file with SA prediction results [converted from .txt]

# Load SA prediction results
df = pd.read_excel(filename, engine='openpyxl')

# Filter for valid SA predictions only
df = df[df['Predicted score'] > 0]

# Extract Specific Activity predictions
predicted_SA = df['Predicted score']

# Extract spectroscopic feature columns
predicted_col_index = df.columns.get_loc('Predicted score')
augmented_data_columns = df.columns[predicted_col_index + 1:].tolist()

# Combine all spectroscopic features into a single list per sample
df['AugmentedData'] = df[augmented_data_columns].apply(lambda row: row.dropna().tolist(), axis=1)

# Define SA performance thresholds
low = 5   # Lower threshold for SA (A/m²-Pt)
high = 25 # Upper threshold for SA (A/m²-Pt)

# Categorize samples based on SA performance
df['Category'] = pd.cut(df['Predicted score'],
                        bins=[-float('inf'), low, high, 30],
                        labels=['Bad Sample', 'Neutral Sample', 'Good Sample'],
                        right=False)

# Separate high and low performing SA samples
good_samples_df = df[df['Category'] == 'Good Sample']
bad_samples_df = df[df['Category'] == 'Bad Sample']

# Create dictionaries mapping SA values to spectroscopic features
good_samples_dict = good_samples_df.set_index('Predicted score')['AugmentedData'].to_dict()
bad_samples_dict = bad_samples_df.set_index('Predicted score')['AugmentedData'].to_dict()

# Calculate mean SA values for each performance category
mean_good_keys = np.mean(list(good_samples_dict.keys()))
mean_bad_keys = np.mean(list(bad_samples_dict.keys()))

# Report sample distribution statistics
print("good count", len(good_samples_dict), "low", low, "high", high)
print("bad count", len(bad_samples_dict), "low", low, "high", high)
# Generate SA distribution plot with performance thresholds
plt.figure(figsize=(10, 6))
sns.histplot(predicted_SA, kde=True, label='Predicted SA')
plt.axvline(mean_good_keys, color='r', linestyle='--', label=f'Mean Good Sample Score: {mean_good_keys:.2f}')
plt.axvline(mean_bad_keys, color='b', linestyle='--', label=f'Mean Bad Sample Score: {mean_bad_keys:.2f}')
plt.xlabel('Predicted SA')
plt.ylabel('Frequency')
plt.title(f'Distribution of Predicted SA')
plt.legend()
plt.savefig(f"{DIR_Data}/distribution_Predicted_SA.png")
plt.show()

def calculate_elementwise_mean(data_dict):
    """Calculate element-wise mean of spectroscopic features across samples.

    Args:
        data_dict: Dictionary mapping SA values to spectroscopic feature arrays

    Returns:
        numpy.ndarray: Mean spectroscopic feature values across all samples
    """
    data_arrays = [np.array(data_list) for data_list in data_dict.values()]
    stacked_data = np.stack(data_arrays)
    mean_data = np.mean(stacked_data, axis=0)
    return mean_data

# Calculate mean spectroscopic signatures for each SA performance category
good_samples_mean = []
if len(good_samples_dict) != 0:
    good_samples_mean = calculate_elementwise_mean(good_samples_dict)
    a = good_samples_mean[:300]  # First 300 features (e.g., EXAFS)

bad_samples_mean = []
if len(bad_samples_dict) != 0:
    bad_samples_mean = calculate_elementwise_mean(bad_samples_dict)
    b = bad_samples_mean[:300]   # First 300 features (e.g., EXAFS)

# Export mean spectroscopic signatures for each performance category
df_good = pd.DataFrame({'Good Samples Mean': good_samples_mean})
df_bad = pd.DataFrame({'Bad Samples Mean': bad_samples_mean})

# Save individual category signatures
df_good.to_excel(f'{DIR_Data}good_samples_mean_II_SA.xlsx', index=False)
df_bad.to_excel(f'{DIR_Data}bad_samples_mean_II_SA.xlsx', index=False)

# Generate comparative plot of mean spectroscopic signatures
plt.figure(figsize=(25, 6))
plt.plot(good_samples_mean, label='Good Samples')
plt.plot(bad_samples_mean, label='Bad Samples')
plt.title(f'Mean Augmented Data for Samples SA')
plt.legend()
plt.savefig(f"{DIR_Data}/Mean_Sample_SA.png")
plt.show()

# Export combined spectroscopic signatures based on available data
if len(bad_samples_mean) > 0 and len(good_samples_mean) > 0:
    output_df = pd.DataFrame({
        'Good Data': good_samples_mean,
        'Bad Data': bad_samples_mean
    })
    output_file_path = f"{DIR_Data}/good_bad_samples_mean_data_SA.xlsx"
    output_df.to_excel(output_file_path, header=False)
elif len(bad_samples_mean) > 0:
    output_df = pd.DataFrame({
        'Bad Data': bad_samples_mean
    })
    output_file_path = f"{DIR_Data}/good_bad_samples_mean_data_SA.xlsx"
    output_df.to_excel(output_file_path, header=False)
elif len(good_samples_mean) > 0:
    output_df = pd.DataFrame({
        'Good Data': good_samples_mean
    })
    output_file_path = f"{DIR_Data}/good_bad_samples_mean_data_SA.xlsx"
    output_df.to_excel(output_file_path, header=False)
else:
    print("Nothing found")
