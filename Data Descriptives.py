import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create a dataframe from the given data and print it for visualization
data = pd.read_csv(filepath_or_buffer='dataset_assignment1.csv', header=0)
print(data)

# Overall data descriptives
print(data.describe())

# Class specific data descriptives
grouped_data = data.groupby('class')
class0_describe = data[data["class"] == 0].drop("class", axis=1).describe()
class1_describe = data[data["class"] == 1].drop("class", axis=1).describe()
print(f'Data Descriptives for class 0:\n{class0_describe}')
print(f'Data Descriptives for class 1:\n{class1_describe}')

# Create a dataframe containg the relevant plot values for x and y
class_0_means = class0_describe.loc['mean']
class_1_means = class1_describe.loc['mean']

y0 = class0_describe.columns.tolist()
y1 = class1_describe.columns.tolist()

# Plotting the two lines on the same graph for comparison
plt.figure(figsize=(10, 6))
plt.plot(class_0_means, label='Class 0')
plt.plot(class_1_means, label='Class 1')
plt.xlabel('Features')
plt.ylabel('Means')
plt.title('Feature means for classes')
plt.legend()

# Create a scatter plot for every possible pair of features
for i in range(len(class0_describe.columns.tolist()) - 1):
    for j in range(i + 1, len(class0_describe.columns.tolist())):
        plt.figure(figsize=(7, 3))

        # Scatter plot for the current pair of features
        plt.scatter(data[data['class'] == 0][class0_describe.columns.tolist()[i]], data[data['class'] == 0][class0_describe.columns.tolist()[j]], label='Class 0', alpha=0.7)
        plt.scatter(data[data['class'] == 1][class0_describe.columns.tolist()[i]], data[data['class'] == 1][class0_describe.columns.tolist()[j]], label='Class 1', alpha=0.7)

        # Fine tuning the scatter plot representation for the user
        plt.xlabel(class0_describe.columns.tolist()[i])
        plt.ylabel(class0_describe.columns.tolist()[j])
        plt.title(f'Scatter Plot of {class0_describe.columns.tolist()[i]} & {class0_describe.columns.tolist()[j]}')
        plt.legend()
        plt.show()