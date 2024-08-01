# NLP Project

This project performs various Natural Language Processing (NLP) tasks, including data preprocessing, visualization, dimensionality reduction, and similarity analysis using a dataset of text data.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Setup and Installation](#setup-and-installation)
3. [Data Preprocessing](#data-preprocessing)
4. [Data Visualization](#data-visualization)
5. [Bag of Words Matrix](#bag-of-words-matrix)
6. [Dimensionality Reduction](#dimensionality-reduction)
7. [Similarity Analysis](#similarity-analysis)
8. [Usage](#usage)
9. [Results](#results)

## Project Structure
├── dataset.csv

├── cleaned_dataset.csv

├── top_30_words.csv

├── train_dataset.csv

├── test_dataset.csv

├── word_frequencies_train.csv

├── words.csv

├── README.md

└── src

└── script.py



## Setup and Installation

 Clone the repository:
    ```sh
    git clone https://github.com/m4skari/linear-algebra-project-NLP.git
    cd linear-algebra-project-NLP
    `````

## Data Preprocessing

The preprocessing step involves cleaning the dataset by removing non-alphanumeric characters, converting text to lowercase, and dropping any missing values.

```python
import pandas as pd
import re

df = pd.read_csv('dataset.csv')

def clean_and_lower(text):
    if isinstance(text, str):
        cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        return cleaned_text.strip()
    return text

df = df.applymap(clean_and_lower)
df.dropna(inplace=True)
df.to_csv('cleaned_dataset.csv', index=False)
```
## Data visualization
Visualize the frequency of the top 30 most common words in the dataset using bar charts and word clouds.

import pandas as pd
from collections import Counter
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud

df = pd.read_csv('cleaned_dataset.csv')

def count_words(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return Counter(words)

combined_text = ' '.join(df.apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1))
word_counts = count_words(combined_text)
most_common_words = word_counts.most_common(30)

df_most_common = pd.DataFrame(most_common_words, columns=['Word', 'Frequency'])
df_most_common.to_csv('top_30_words.csv', index=False)

plt.figure(figsize=(12, 8))
plt.bar(df_most_common['Word'], df_most_common['Frequency'], color='skyblue')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 30 Most Common Words')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Most Common Words')
plt.show()
```
