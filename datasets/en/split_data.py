import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV file from Google Drive
file_path = '/content/drive/My Drive/globemail data/merged_file.csv'  # Update the path to your CSV file
df = pd.read_csv(file_path)

# Create a new column 'sentence_length' with the length of each sentence
df['sentence_length'] = df['Original'].apply(lambda x: len(x.split()))

# Categorize sentences based on their length
bins = [12, 22, 32, 42, 51]  # define the bins for categorizing lengths
labels = ['13-22', '23-32', '33-42', '43-51']
df['length_category'] = pd.cut(df['sentence_length'], bins=bins, labels=labels, right=True)

# Split data by category
def stratified_split(df, category, train_size, test_size, random_state=42):
    train, test = train_test_split(
        df[df['length_category'] == category], 
        train_size=train_size, 
        test_size=test_size, 
        random_state=random_state, 
        shuffle=True
    )
    return train, test

# Define proportions
valid_proportions = {'13-22': 224, '23-32': 224, '33-42': 44, '43-51': 8}
test_proportions = {'13-22': 897, '23-32': 894, '33-42': 176, '43-51': 33}

valid_dfs = []
test_dfs = []

for category in labels:
    valid_train, valid_test = stratified_split(
        df, category, 
        train_size=valid_proportions[category], 
        test_size=test_proportions[category]
    )
    valid_dfs.append(valid_train)
    test_dfs.append(valid_test)

# Combine all categories into final DataFrames
valid_df = pd.concat(valid_dfs)
test_df = pd.concat(test_dfs)

# Save the split datasets to CSV
valid_df.to_csv('/content/drive/My Drive/globemail data/valid_data_new.csv', index=False)
test_df.to_csv('/content/drive/My Drive/globemail data/test_data_new.csv', index=False)

print('Validation and test datasets created successfully!')