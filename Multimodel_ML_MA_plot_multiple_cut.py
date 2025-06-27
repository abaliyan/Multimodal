import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

customsequence = ('EXAFS_K2', 'XRD_2_P', 'XANES', 'PDF', 'HAXPES_VB', 'SAXS', 'HAXPES_Pt3d', 'HAXPES_Pt4f')
DIR_Data = "models_test"
filename = f"{DIR_Data}/Aug_Predicted_Octa_MA.xlsx"

df = pd.read_excel(filename, engine='openpyxl')

df = df[df['Predicted score'] > 0]

# Extract the predicted scores
predicted_MA = df['Predicted score']

# Define a list of augmented data column names (adjust these names as needed)
predicted_col_index = df.columns.get_loc('Predicted score')
augmented_data_columns = df.columns[predicted_col_index + 1:].tolist()

# Create a new column that combines all augmented data into a single list
df['AugmentedData'] = df[augmented_data_columns].apply(lambda row: row.dropna().tolist(), axis=1)

# Define the ranges for categorization
ranges = {
    "200-250": (200, 250),
    "250-300": (250, 300),
    "300-350": (300, 350),
    "500-600": (500, 600),
    "1000-1050": (1000, 1050),
    "1050-1100": (1050, 1100),
    "1100-1150": (1100, 1150),
    "1150-1200": (1150, 1200),
    "1200-1250": (1200, 1250),
    "1200-1300": (1200, 1300),
    "1300-1350": (1300, 1350),
    "1350-1400": (1350, 1400),
    "1400-1450": (1400, 1450),
    "1450-1500": (1450, 1500),
    "1500-1600": (1500, 1600),
    "1800-1900": (1800, 1900),
    "≥2100": (2100, float("inf"))
}

def categorize(score):
    matched_categories = []
    for category, (low, high) in ranges.items():
        if low <= score < high:
            matched_categories.append(category)
    return matched_categories if matched_categories else None

# Categorize scores
df['Category'] = df['Predicted score'].apply(categorize)

# Explode categories to handle multiple matches per score
df = df.explode('Category')

# Plot histograms for each category
plt.figure(figsize=(12, 8))
for category in ranges.keys():
    category_scores = df[df['Category'] == category]['Predicted score']
    sns.histplot(category_scores, kde=True, label=f'{category}', bins=30)

plt.title('Histograms of Predicted Scores by Category')
plt.xlabel('Predicted Score')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Initialize dictionaries to store data
category_data = {}
category_means = {}

# Process each range
for category, (low, high) in ranges.items():
    category_df = df[df['Category'] == category]
    category_dict = category_df.set_index('Predicted score')['AugmentedData'].to_dict()
    
    print("good count", len(category_dict), "low", low, "high", high)
    # Calculate mean if there is data
    if len(category_dict) > 0:
        category_mean = np.mean(np.stack([np.array(v) for v in category_dict.values()]), axis=0)
    else:
        category_mean = []

    # Store results
    category_data[category] = category_dict
    category_means[category] = category_mean

    # Save to Excel
    category_df = pd.DataFrame({f'{category} Samples Mean': category_mean})
    category_df.to_excel(f'{category.lower().replace("-", "_")}_samples_mean_{ranges[category][0]}_{ranges[category][1] if ranges[category][1] != float("inf") else "inf"}.xlsx', index=False)

# Plot the mean augmented data for all ranges
plt.figure(figsize=(25, 6))
for category, mean_data in category_means.items():
    if len(mean_data) > 0:
        plt.plot(mean_data, label=f'{category} Samples Mean')

plt.title(f'Mean Augmented Data for Predicted MA')
plt.legend()
plt.savefig(f"{DIR_Data}/MA_multiple_cut_Mean_Sample_all_ranges{ranges['1300, 1350']}.png")
plt.show()

# Save all mean augmented data to a single Excel file
output_data = {}
for category, mean_data in category_means.items():
    output_data[category] = mean_data

max_len = max(len(v) for v in output_data.values())
output_df = pd.DataFrame({k: np.pad(v, (0, max_len - len(v)), constant_values=np.nan) for k, v in output_data.items()})
output_file_path = f"{DIR_Data}/MA_multiple_cut_all_ranges_mean_data{ranges["1300, 1350"]}.xlsx"
output_df.to_excel(output_file_path, index=False)