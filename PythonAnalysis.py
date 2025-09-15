# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Set style for better looking plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Task 1: Load and Explore the Dataset
print("Task 1: Load and Explore the Dataset")
print("="*50)

# Load the Iris dataset from sklearn
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Display the first few rows
print("First 5 rows of the dataset:")
print(iris_df.head())
print()

# Explore the structure
print("Dataset info:")
print(iris_df.info())
print()

# Check for missing values
print("Missing values in each column:")
print(iris_df.isnull().sum())
print()

# Since there are no missing values, no cleaning is needed
print("No missing values found. Dataset is clean.")
print()

# Task 2: Basic Data Analysis
print("Task 2: Basic Data Analysis")
print("="*50)

# Basic statistics
print("Basic statistics of numerical columns:")
print(iris_df.describe())
print()

# Group by species and compute mean of numerical columns
print("Mean values by species:")
species_group = iris_df.groupby('species').mean()
print(species_group)
print()

# Identify patterns
print("Observations:")
print("- Setosa has the smallest petal dimensions but similar sepal width to Versicolor")
print("- Virginica has the largest petal dimensions on average")
print("- Sepal length increases from Setosa to Versicolor to Virginica")
print()

# Task 3: Data Visualization
print("Task 3: Data Visualization")
print("="*50)

# Create a figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Iris Dataset Analysis', fontsize=16)

# 1. Line chart - Using sepal length as a pseudo-time series
axes[0, 0].plot(iris_df.index, iris_df['sepal length (cm)'], 
                label='Sepal Length', color='blue')
axes[0, 0].plot(iris_df.index, iris_df['petal length (cm)'], 
                label='Petal Length', color='red')
axes[0, 0].set_title('Trend of Sepal and Petal Length')
axes[0, 0].set_xlabel('Observation Index')
axes[0, 0].set_ylabel('Length (cm)')
axes[0, 0].legend()

# 2. Bar chart - Average sepal length by species
species_means = iris_df.groupby('species').mean()
axes[0, 1].bar(species_means.index, species_means['sepal length (cm)'],
               color=['skyblue', 'lightcoral', 'lightgreen'])
axes[0, 1].set_title('Average Sepal Length by Species')
axes[0, 1].set_xlabel('Species')
axes[0, 1].set_ylabel('Sepal Length (cm)')

# 3. Histogram - Distribution of sepal length
axes[1, 0].hist(iris_df['sepal length (cm)'], bins=15, 
                color='lightblue', edgecolor='black')
axes[1, 0].set_title('Distribution of Sepal Length')
axes[1, 0].set_xlabel('Sepal Length (cm)')
axes[1, 0].set_ylabel('Frequency')

# 4. Scatter plot - Sepal length vs petal length, colored by species
colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
for species, color in colors.items():
    species_data = iris_df[iris_df['species'] == species]
    axes[1, 1].scatter(species_data['sepal length (cm)'], 
                      species_data['petal length (cm)'], 
                      color=color, label=species, alpha=0.7)
axes[1, 1].set_title('Sepal Length vs Petal Length')
axes[1, 1].set_xlabel('Sepal Length (cm)')
axes[1, 1].set_ylabel('Petal Length (cm)')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

# Additional insights
print("Additional Insights from Visualizations:")
print("- Setosa flowers (red) are clearly separated from the other species in the scatter plot")
print("- Versicolor and Virginica have some overlap but are mostly distinguishable")
print("- The distribution of sepal length appears to be bimodal, reflecting the different species")
print("- The bar chart clearly shows Virginica has the longest average sepal length")