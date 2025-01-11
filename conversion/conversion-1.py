import pandas as pd

# Load the first dataset (Offensive and Hate Speech Detection)
df1 = pd.read_csv("labeled_data.csv")  # Replace with the actual path to dataset1

# Load the second dataset (Hate Speech Detection and Severity)
df2 = pd.read_csv("hate.csv")  

# Function to classify based on toxicity and severity
def classify_toxicity(row):
    # Priority 1: Check for insults, which are considered hate speech
    if row['insult'] == 1:
        return 'hate_speech'

    # Priority 2: Check for severe toxicity (strongly offensive)
    if row['severe_toxic'] == 1:
        return 'offensive'
    
    # Priority 3: Check for general toxicity (offensive)
    if row['toxic'] == 1:
        return 'offensive'

    # Priority 4: Check for offensive content (obscene, threat, insult)
    if row['obscene'] == 1 or row['threat'] == 1:
        return 'offensive'

    # Priority 5: Check for identity hate
    if row['identity_hate'] == 1:
        return 'hate_speech'
    
    # Priority 6: If toxic and identity hate together, classify as hate speech
    if row['toxic'] == 1 and row['identity_hate'] == 1:
        return 'hate_speech'
    
    # Priority 7: If all toxicity columns are 0, classify as neither
    if row['toxic'] == 0 and row['severe_toxic'] == 0 and row['obscene'] == 0 and row['threat'] == 0 and row['insult'] == 0 and row['identity_hate'] == 0:
        return 'neither'
    
    # Fallback to 'neither'
    return 'neither'

# Apply the classification function to the second dataset
df2['combined_label'] = df2.apply(classify_toxicity, axis=1)

# Now merge the two datasets based on some common attribute (assuming index or tweet ID alignment)
# Here, we'll use an outer join to align them
df_combined = pd.merge(df1, df2, how='outer', left_index=True, right_index=True, suffixes=('_df1', '_df2'))

# The merged dataset will now have 'combined_label' from df2, which classifies the toxicity
# We can update the final label based on the combined labels from both datasets
df_combined['final_label'] = df_combined['combined_label'].fillna(df_combined['combined_label_df1'])

# Adjusting the final label with more refined rules (example of rule-based label prioritization)
df_combined['final_label'] = df_combined.apply(
    lambda row: 'hate_speech' if row['combined_label'] == 'hate_speech' or row['combined_label_df1'] == 'hate_speech' else row['final_label'],
    axis=1
)

# Final priority rules for the combined dataset
df_combined['final_label'] = df_combined.apply(
    lambda row: 'offensive' if (row['combined_label'] == 'offensive' or row['combined_label_df1'] == 'offensive') and row['final_label'] != 'hate_speech' else row['final_label'],
    axis=1
)

# If both datasets say 'neither', set it as neither, otherwise mark as offensive or hate_speech
df_combined['final_label'] = df_combined.apply(
    lambda row: 'neither' if row['final_label'] == 'neither' and row['combined_label_df1'] == 'neither' else row['final_label'],
    axis=1
)

# Remove any rows where both the labels are redundant or conflicting (e.g., both datasets say 'offensive' or 'hate_speech')
df_combined = df_combined.drop_duplicates(subset=['final_label'])

# Save the combined dataset
df_combined.to_csv("combined_dataset_v3.csv", index=False)

print(df_combined.head())
