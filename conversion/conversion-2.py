import pandas as pd

data = pd.read_csv("aa3.csv")

# Modify the 'class' column
data['class'] = data['class'].apply(lambda x: 0 if x in [0, 1] else 1)

# Save the updated dataset with the modified 'class' column
data.to_csv('labeled_data.csv', index=False)
print(data.head())
