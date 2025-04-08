##Thinking Level	Description
## Basic	Transform, clean, encode
## Intermediate	Rolling stats, binary interactions, lag features
## Expert	Model-aware design, domain logic, redundancy handling, interaction crafting,
# temporal memory, automated selection (SHAP, mutual info)

##Basic -
##* Handle missing values
##* Basic temporal feature extraction (day, weekday, month)
##* One-hot encode categoricals
##* Combine binary indicators (holiday/weekend)
##* Drop irrelevant columns
##* Normalize features


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

##read raw dataset
data = pd.read_csv('synthetic_email_spam.csv')

print(data.info)
print(data.columns)
print(data.head(5))
print(data.size)


numeric_features = data.select_dtypes(include=["int64","float64"])


##compute correlation matric
corr_matrix = numeric_features.corr()

# ##visualize correlation matrix -
# plt.figure(figsize=(10, 6))
# sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
# plt.title("Feature Correlation Matrix")
# plt.show()

##Here’s how to programmatically remove one feature from highly correlated pairs (e.g., correlation > 0.85):

##removing correlation redundency from datasets
# Set a threshold for high correlation
threshold = 0.85

# Get upper triangle of correlation matrix
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
# Find features with correlation greater than threshold
to_drop = [column for column in upper_tri.columns if any(upper_tri[column].abs() > threshold)]


print("Highly correlated features to drop:", to_drop)


# Drop them from the dataframe
data = data.drop(columns=to_drop)

print("Remaining features:", data.columns.tolist())

##now in our net steps -
#3we will walk through expert level feature engineering
##like identifiyng features which can improve dataset
##perform feature interactions

##Combine related indicators:

##suspicious_keywords = sum of ['viagra', 'free', 'buy now', 'click here']
##suspicious_sender = 1 if email is from unknown/random domain
##num_special_chars = count of !, $, %, etc.
##normalized link_density = links per 100 words

n = len(data)

contains_viagra = np.random.choice([1,0],n)
contains_click = np.random.choice([1,0],n)

##add into dataframe
data['contains_viagra']=contains_viagra
data['contains_click']=contains_click
data['char_count'] = data['num_special_chars'] + np.random.randint(10, 1000, n)
data['word_count'] = data['char_count'] - np.random.randint(1,10,n)

##new features additions
data['suspicious_keywords'] = data[['contains_viagra', 'contains_free', 'contains_click','contains_offer']].sum(axis=1)
data['special_char_ratio'] = data['num_special_chars'] / (data['char_count'] + 1)
data['link_density'] = data['num_links'] / (data['word_count'] + 1)

print(data.columns)
print(data.info)
print(data.head(5))

##non-linear transformation

##try to reduce skewness on columns
##apply log1g

data['log_char_count'] = np.log1p(data['char_count'])
print(data['log_char_count'].head(5))

##use binerization if features have threshold behaviour
data['is_very_long'] = (data['char_count'] > 500).astype(int)
print(data['is_very_long'].head(5))

data['log_word_count'] = np.log1p(data['word_count'])
print(data['log_word_count'].head(5))

##now perform the feature interactions
##this will help mimic no linearity into datasets

data['spam_score'] = data['num_links']*data['suspicious_keywords']
data['special_link_interaction'] = data['link_density'] * data['special_char_ratio']


print(data.head(5))

print(data.columns)



