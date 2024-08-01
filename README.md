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
