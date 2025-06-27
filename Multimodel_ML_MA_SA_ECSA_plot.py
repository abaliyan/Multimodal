import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


customsequence = ('EXAFS_K2', 'XRD_2_P', 'XANES', 'PDF', 'HAXPES_VB', 'SAXS', 'HAXPES_Pt3d', 'HAXPES_Pt4f')
DIR_Data = "models_test"
filename = f"{DIR_Data}/Aug_Predicted_Octa_MA.xlsx"

df = pd.read_excel(filename, engine='openpyxl')

# Keep only rows with valid predicted values
df = df[df['Predicted'] > 0]

# Extract your three targets
predicted_MA = df['Predicted']
predicted_SA = df['score']
predicted_ECSA = df['Augmented']

# Define the augmented data columns (everything after 'Augmented')
augmented_col_index = df.columns.get_loc('Augmented')
augmented_data_columns = df.columns[augmented_col_index + 1:].tolist()

# Create a new column with all augmented data as a list
df['AugmentedData'] = df[augmented_data_columns].apply(lambda row: row.dropna().tolist(), axis=1)

# Categorize samples by MA
low = 50
high = 1700
df['Category'] = pd.cut(
    df['Predicted'],
    bins=[-float('inf'), low, high, float('inf')],
    labels=['Bad Sample', 'Neutral Sample', 'Good Sample'],
    right=False
)

# Split into good and bad samples
good_samples_df = df[df['Category'] == 'Good Sample']
bad_samples_df = df[df['Category'] == 'Bad Sample']

good_samples_dict = good_samples_df.set_index('Predicted')['AugmentedData'].to_dict()
bad_samples_dict = bad_samples_df.set_index('Predicted')['AugmentedData'].to_dict()

mean_good_MA = np.mean(list(good_samples_dict.keys()))
mean_bad_MA = np.mean(list(bad_samples_dict.keys()))
mean_good_SA = good_samples_df['score'].mean()
mean_bad_SA = bad_samples_df['score'].mean()
mean_good_ECSA = good_samples_df['Augmented'].mean()
mean_bad_ECSA = bad_samples_df['Augmented'].mean()

print("good count", len(good_samples_dict), "low", low, "high", high)
print("bad count", len(bad_samples_dict), "low", low, "high", high)

print(f"Mean SA Good: {mean_good_SA:.2f}")
print(f"Mean SA Bad: {mean_bad_SA:.2f}")
print(f"Mean ECSA Good: {mean_good_ECSA:.2f}")
print(f"Mean ECSA Bad: {mean_bad_ECSA:.2f}")

# -------- SA Distribution --------
plt.figure(figsize=(10, 6))
sns.histplot(predicted_SA, kde=True, label='SA', binwidth=3)
plt.axvline(mean_good_SA, color='r', linestyle='--', label=f'Mean Good SA: {mean_good_SA:.2f}')
plt.axvline(mean_bad_SA, color='b', linestyle='--', label=f'Mean Bad SA: {mean_bad_SA:.2f}')
plt.xlabel('SA')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted SA')
plt.xlim(0, 50)
plt.legend()
plt.savefig(f"{DIR_Data}/distribution_Predicted_MASAECSA_SA.png")
plt.show()

# -------- MA Distribution --------
plt.figure(figsize=(10, 6))
sns.histplot(predicted_MA, kde=True, label='MA')
plt.axvline(mean_good_MA, color='r', linestyle='--', label=f'Mean Good MA: {mean_good_MA:.2f}')
plt.axvline(mean_bad_MA, color='b', linestyle='--', label=f'Mean Bad MA: {mean_bad_MA:.2f}')
plt.xlabel('MA')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted MA')
plt.legend()
plt.savefig(f"{DIR_Data}/distribution_Predicted_MASAECSA_MA.png")
plt.show()

# -------- ECSA Distribution --------
plt.figure(figsize=(10, 6))
sns.histplot(predicted_ECSA, kde=True, label='ECSA')
plt.axvline(mean_good_ECSA, color='r', linestyle='--', label=f'Mean Good ECSA: {mean_good_ECSA:.2f}')
plt.axvline(mean_bad_ECSA, color='b', linestyle='--', label=f'Mean Bad ECSA: {mean_bad_ECSA:.2f}')
plt.xlabel('ECSA')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted ECSA')
plt.legend()
plt.savefig(f"{DIR_Data}/distribution_Predicted_MASAECSA_ECSA.png")
plt.show()

# -------- Compute element-wise mean of augmented data --------
def calculate_elementwise_mean(data_dict):
    data_arrays = [np.array(v) for v in data_dict.values()]
    stacked = np.stack(data_arrays)
    return np.mean(stacked, axis=0)

good_samples_mean = calculate_elementwise_mean(good_samples_dict)
bad_samples_mean = calculate_elementwise_mean(bad_samples_dict)

# Save individual means
df_good = pd.DataFrame({'Good Samples Mean': good_samples_mean})
df_bad = pd.DataFrame({'Bad Samples Mean': bad_samples_mean})
df_good.to_excel(f'{DIR_Data}/good_samples_mean_II_ECSA_MA_SA.xlsx', index=False)
df_bad.to_excel(f'{DIR_Data}/bad_samples_mean_II_ECSA_MA_SA.xlsx', index=False)

# -------- Plot Mean Augmented Data --------
plt.figure(figsize=(25, 6))
plt.plot(good_samples_mean, label='Good')
plt.plot(bad_samples_mean, label='Bad')
plt.title('Mean Augmented Data')
plt.legend()
plt.savefig(f"{DIR_Data}/Mean_Sample_MASAECSA.png")
plt.show()

# Save combined Excel
output_df = pd.DataFrame({
    'Good Data': good_samples_mean,
    'Bad Data': bad_samples_mean
})
output_df.to_excel(f'{DIR_Data}/good_bad_samples_mean_data_MASAECSA.xlsx', index=False)
