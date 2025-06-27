import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

customsequence = ('EXAFS_K2', 'XRD_2_P', 'XANES', 'PDF', 'HAXPES_VB', 'SAXS', 'HAXPES_Pt3d', 'HAXPES_Pt4f')
DIR_Data = "D:/MM_final/models_test"
filename = f"{DIR_Data}/Aug_Predicted_Octa_MA_SA.xlsx"
df = pd.read_excel(filename, engine='openpyxl')

# Remove any rows where predicted value is not positive
df = df[df['Predicted'] > 0]

# Extract MA and SA
predicted_MA = df['Predicted']
predicted_SA = df['score']

low = 50
high = 1700

# Get the list of augmented data columns
# We assume AugData_* columns come **after** 'score'
score_col_index = df.columns.get_loc('score')
augmented_data_columns = df.columns[score_col_index + 1:].tolist()

# Combine augmented data into one list per row
df['AugmentedData'] = df[augmented_data_columns].apply(lambda row: row.dropna().tolist(), axis=1)

# Categorize MA into bins
df['Category'] = pd.cut(
    df['Predicted'],
    bins=[-float('inf'), low, high, float('inf')],
    labels=['Bad Sample', 'Neutral Sample', 'Good Sample']
)

# Separate data into good and bad
good_samples_df = df[df['Category'] == 'Good Sample']
bad_samples_df = df[df['Category'] == 'Bad Sample']

# Dictionaries mapping MA score -> augmented data
good_samples_dict = good_samples_df.set_index('Predicted')['AugmentedData'].to_dict()
bad_samples_dict = bad_samples_df.set_index('Predicted')['AugmentedData'].to_dict()

# Means for vertical lines
mean_good_MA = np.mean(list(good_samples_dict.keys()))
mean_bad_MA = np.mean(list(bad_samples_dict.keys()))
mean_good_SA = good_samples_df['score'].mean()
mean_bad_SA = bad_samples_df['score'].mean()

print("Good count", len(good_samples_dict), "low", low, "high", high)
print("Bad count", len(bad_samples_dict), "low", low, "high", high)

# -------- PLOT SA ---------
plt.figure(figsize=(10, 6))
sns.histplot(predicted_SA, kde=True, label='SA Data', binwidth=3)
plt.axvline(mean_good_SA, color='r', linestyle='--', label=f'Mean Good Sample SA: {mean_good_SA:.2f}')
plt.axvline(mean_bad_SA, color='b', linestyle='--', label=f'Mean Bad Sample SA: {mean_bad_SA:.2f}')
plt.xlabel('Predicted SA')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted SA')
plt.xlim(0, 50)
plt.legend()
plt.savefig(f"{DIR_Data}/distribution_Predicted_MASA_SA.png")
plt.show()

# -------- PLOT MA ---------
plt.figure(figsize=(10, 6))
sns.histplot(predicted_MA, kde=True, label='MA Data')
plt.axvline(mean_good_MA, color='r', linestyle='--', label=f'Mean Good Sample MA: {mean_good_MA:.2f}')
plt.axvline(mean_bad_MA, color='b', linestyle='--', label=f'Mean Bad Sample MA: {mean_bad_MA:.2f}')
plt.xlabel('Predicted MA')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted MA')
plt.legend()
plt.savefig(f"{DIR_Data}/distribution_Predicted_MASA_MA.png")
plt.show()

# -------- Element-wise Means ---------
def calculate_elementwise_mean(data_dict):
    data_arrays = [np.array(lst) for lst in data_dict.values()]
    stacked = np.stack(data_arrays)
    return np.mean(stacked, axis=0)

good_samples_mean = calculate_elementwise_mean(good_samples_dict)
bad_samples_mean = calculate_elementwise_mean(bad_samples_dict)

# Plot
plt.figure(figsize=(25, 6))
plt.plot(good_samples_mean, label='Good')
plt.plot(bad_samples_mean, label='Bad')
plt.title('Mean Augmented Data for Samples')
plt.legend()
plt.savefig(f"{DIR_Data}/Mean_Sample_MA_SA.png")
plt.show()

# Save to Excel
output_df = pd.DataFrame({
    'Good Data': good_samples_mean,
    'Bad Data': bad_samples_mean
})
output_df.to_excel(f"{DIR_Data}/good_bad_samples_mean_data_MA_SA.xlsx", index=False)
