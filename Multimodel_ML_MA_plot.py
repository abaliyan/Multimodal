"""Mass Activity (MA) Analysis and Visualization Script.

This script analyzes predicted Mass Activity values from multi-modal
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
filename = r""  # Path to Excel file with MA prediction results [converted from .txt]
# Load MA prediction results
df = pd.read_excel(filename, engine='openpyxl')

# Filter for valid MA predictions only
df = df[df['Predicted score'] > 0]

# Extract Mass Activity predictions
predicted_MA = df['Predicted score']

# Extract spectroscopic feature columns
predicted_col_index = df.columns.get_loc('Predicted score')
augmented_data_columns = df.columns[predicted_col_index + 1:].tolist()

# Combine all spectroscopic features into a single list per sample
df['AugmentedData'] = df[augmented_data_columns].apply(lambda row: row.dropna().tolist(), axis=1)

# Define MA performance thresholds
low = 10    # Lower threshold for MA (A/g-Pt)
high = 1750 # Upper threshold for MA (A/g-Pt)

# Categorize samples based on MA performance
df['Category'] = pd.cut(df['Predicted score'],
                        bins=[-float('inf'), low, high, 2300],
                        labels=['Bad Sample', 'Neutral Sample', 'Good Sample'],
                        right=False)

# Separate high and low performing MA samples
good_samples_df = df[df['Category'] == 'Good Sample']
bad_samples_df = df[df['Category'] == 'Bad Sample']

# Create dictionaries mapping MA values to spectroscopic features
good_samples_dict = good_samples_df.set_index('Predicted score')['AugmentedData'].to_dict()
bad_samples_dict = bad_samples_df.set_index('Predicted score')['AugmentedData'].to_dict()

mean_good_keys = np.mean(list(good_samples_dict.keys()))
mean_bad_keys = np.mean(list(bad_samples_dict.keys()))

# Report sample distribution statistics
print("good count", len(good_samples_dict), "low", low, "high", high)
print("bad count", len(bad_samples_dict), "low", low, "high", high)

# Generate MA distribution plot with performance thresholds
plt.figure(figsize=(10, 6))
sns.histplot(predicted_MA, kde=True, label='Predicted MA', bins=30)
plt.axvline(mean_good_keys, color='r', linestyle='--', label=f'Mean Good Sample Score: {mean_good_keys:.2f}')
plt.axvline(mean_bad_keys, color='b', linestyle='--', label=f'Mean Bad Sample Score: {mean_bad_keys:.2f}')
plt.xlabel('Predicted MA')
plt.ylabel('Frequency')
plt.title(f'Distribution of Predicted MA')
plt.legend()
plt.savefig(f"distribution_Predicted_MA.png")
plt.show()

def calculate_elementwise_mean(data_dict):
    """Calculate element-wise mean of spectroscopic features across samples.

    Args:
        data_dict: Dictionary mapping MA values to spectroscopic feature arrays

    Returns:
        numpy.ndarray: Mean spectroscopic feature values across all samples
    """
    data_arrays = [np.array(data_list) for data_list in data_dict.values()]
    stacked_data = np.stack(data_arrays)
    mean_data = np.mean(stacked_data, axis=0)
    return mean_data

# Calculate mean spectroscopic signatures for each MA performance category
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
df_good.to_excel(f'{DIR_Data}good_samples_mean_II_MA.xlsx', index=False)
df_bad.to_excel(f'{DIR_Data}bad_samples_mean_II_MA.xlsx', index=False)

# Generate comparative plot of mean spectroscopic signatures
plt.figure(figsize=(25, 6))
plt.plot(good_samples_mean, label='Good Samples')
plt.plot(bad_samples_mean, label='Bad Samples')
plt.title(f'Mean Augmented Data for Samples MA')
plt.legend()
plt.savefig(f"{DIR_Data}/Mean_Sample_MA.png")
plt.show()

# Export combined spectroscopic signatures based on available data
if len(bad_samples_mean) > 0 and len(good_samples_mean) > 0:
    output_df = pd.DataFrame({
        'Good Data': good_samples_mean,
        'Bad Data': bad_samples_mean
    })
    output_file_path = f"{DIR_Data}/good_bad_samples_mean_data_MA.xlsx"
    output_df.to_excel(output_file_path, header=False)
elif len(bad_samples_mean) > 0:
    output_df = pd.DataFrame({
        'Bad Data': bad_samples_mean
    })
    output_file_path = f"{DIR_Data}/good_bad_samples_mean_data_MA.xlsx"
    output_df.to_excel(output_file_path, header=False)
elif len(good_samples_mean) > 0:
    output_df = pd.DataFrame({
        'Good Data': good_samples_mean
    })
    output_file_path = f"{DIR_Data}/good_bad_samples_mean_data_MA.xlsx"
    output_df.to_excel(output_file_path, header=False)
else:
    print("Nothing found")
