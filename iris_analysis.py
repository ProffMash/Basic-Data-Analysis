# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset

def load_and_explore_data():
    """
    Load the Iris dataset and perform initial exploration
    """
    try:
        # Load the Iris dataset
        iris = load_iris()
        
        # Create a DataFrame
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['species'] = iris.target
        df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        
        # Display first few rows
        print("First 5 rows of the dataset:")
        print(df.head())
        print("\n")
        
        # Explore structure
        print("Dataset info:")
        print(df.info())
        print("\n")
        
        # Check for missing values
        print("Missing values per column:")
        print(df.isnull().sum())
        print("\n")
        
        # Since there are no missing values in this clean dataset, we'll demonstrate cleaning
        # by creating a sample with missing values and cleaning it
        print("Demonstrating data cleaning with a sample copy:")
        df_sample = df.copy()
        df_sample.iloc[10:15, 1] = None  # Introduce some missing values
        
        print("Missing values before cleaning:")
        print(df_sample.isnull().sum())
        
        # Clean by filling with mean (or could drop)
        mean_value = df_sample['sepal width (cm)'].mean()
        df_sample['sepal width (cm)'].fillna(mean_value, inplace=True)
        
        print("\nMissing values after cleaning:")
        print(df_sample.isnull().sum())
        print("\n")
        
        return df
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Task 2: Basic Data Analysis

def perform_data_analysis(df):
    """
    Perform basic statistical analysis on the dataset
    """
    try:
        # Basic statistics
        print("Basic statistics for numerical columns:")
        print(df.describe())
        print("\n")
        
        # Group by species and compute mean
        print("Mean measurements by species:")
        print(df.groupby('species').mean())
        print("\n")
        
        # Additional interesting findings
        print("Additional findings:")
        print("- Setosa has the smallest petal measurements")
        print("- Virginica has the largest petal measurements")
        print("- Versicolor is in between for most measurements")
        print("\n")
        
    except Exception as e:
        print(f"An error occurred during analysis: {e}")

# Task 3: Data Visualization

def create_visualizations(df):
    """
    Create various visualizations of the data
    """
    try:
        # Set style for better looking plots
        sns.set(style="whitegrid")
        
        # 1. Line chart (though Iris isn't time series, we'll use index as pseudo-time)
        plt.figure(figsize=(10, 5))
        plt.plot(df.index[:50], df['sepal length (cm)'][:50], label='Sepal Length')
        plt.plot(df.index[:50], df['petal length (cm)'][:50], label='Petal Length')
        plt.title('Trend of Sepal and Petal Length (First 50 Samples)')
        plt.xlabel('Sample Index')
        plt.ylabel('Length (cm)')
        plt.legend()
        plt.show()
        
        # 2. Bar chart - average sepal length by species
        plt.figure(figsize=(8, 5))
        df.groupby('species')['sepal length (cm)'].mean().plot(kind='bar', color=['skyblue', 'salmon', 'lightgreen'])
        plt.title('Average Sepal Length by Species')
        plt.xlabel('Species')
        plt.ylabel('Average Sepal Length (cm)')
        plt.xticks(rotation=0)
        plt.show()
        
        # 3. Histogram - distribution of petal width
        plt.figure(figsize=(8, 5))
        sns.histplot(data=df, x='petal width (cm)', bins=15, kde=True, hue='species', multiple='stack')
        plt.title('Distribution of Petal Width by Species')
        plt.xlabel('Petal Width (cm)')
        plt.ylabel('Frequency')
        plt.show()
        
        # 4. Scatter plot - sepal length vs petal length
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species', style='species', s=100)
        plt.title('Sepal Length vs Petal Length by Species')
        plt.xlabel('Sepal Length (cm)')
        plt.ylabel('Petal Length (cm)')
        plt.legend(title='Species')
        plt.show()
        
        # Bonus: Pairplot to show all relationships
        plt.figure(figsize=(10, 8))
        sns.pairplot(df, hue='species', height=2)
        plt.suptitle('Pairwise Relationships in Iris Dataset', y=1.02)
        plt.show()
        
    except Exception as e:
        print(f"An error occurred during visualization: {e}")

# Main execution
if __name__ == "__main__":
    print("=== Task 1: Load and Explore Dataset ===")
    iris_df = load_and_explore_data()
    
    if iris_df is not None:
        print("\n=== Task 2: Basic Data Analysis ===")
        perform_data_analysis(iris_df)
        
        print("\n=== Task 3: Data Visualization ===")
        create_visualizations(iris_df)