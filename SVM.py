#----SUPPORT VECTOR MACHINE----
# Importing libraries and modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

#  Initiliazing datasets into a training and testing split of 80/20
data = pd.read_csv(filepath_or_buffer='dataset_assignment1.csv', header=0)
input_x = data.drop('class', axis=1)
ground_truth = data['class']
grouped_data = data.groupby('class')
data_class1 = data[data['class'] == 1].drop('class', axis=1)
data_class0 = data[data['class'] == 0].drop('class', axis=1)
x_train, x_test, y_train, y_test = train_test_split(input_x, ground_truth, test_size=0.2, random_state=42)

# Initiliazing the model and feeding in the training dataset
model = SVC(random_state=42)
model.fit(x_train, y_train)

# Evaluating the model on the test dataset using default hyperparameters
y_predict = model.predict(x_test)
print('Using default parameters:\n')
print('Accuracy: %.2f' % metrics.accuracy_score(y_test, y_predict))
print('Precision: %.2f' % metrics.precision_score(y_test, y_predict))
print('Recall: %.2f' % metrics.recall_score(y_test, y_predict))
print('F1 Score: %.2f' % metrics.f1_score(y_test, y_predict))

# Finding the best combination of hyperparamters using sklearn's grid search. Using numpy to create lists for hyperparamters
params_for_grid = {'C':(np.arange(1.0,20.0)).tolist()[0::2],
              'degree':np.arange(0,12).tolist(),
              'gamma' : ['scale', 'auto'],
              'kernel':['linear', 'poly', 'rbf', 'sigmoid']
              }

# Creating an instance of the grid search class, setting cv to 5 in order to use a 5-fold cross validation
grid_search = GridSearchCV(SVC(random_state=42), params_for_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Refitting the model using the updated hypermarameters
grid_cv = grid_search.fit(x_train, y_train)
best_params = grid_search.best_params_
new_model = SVC(**best_params, random_state=42)
new_model.fit(x_train, y_train)
y_predict_new = new_model.predict(x_test)

results = pd.DataFrame(grid_cv.cv_results_)

# Printing model scores and confusion matrix
print(f'\nUpdated parameters:\n')
for key, value in best_params.items():
    print(f"{key}: {value}")

confusion_matrix = metrics.confusion_matrix(y_test.tolist(), y_predict_new)
print('\nAccuracy: %.2f' % metrics.accuracy_score(y_test, y_predict_new))
print('Precision: %.2f' % metrics.precision_score(y_test, y_predict_new))
print('Recall: %.2f' % metrics.recall_score(y_test, y_predict_new))
print('F1 Score: %.2f' % metrics.f1_score(y_test, y_predict_new))

# Illustrarting the confusion matrix using seaborn
print(sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Class 0", "Class 1"],
            yticklabels=["Class 0", "Class 1"]))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# visualizing the results of hyperparamter tuning
heatmap_data = results.pivot_table(index='param_degree', columns='param_C', values='mean_test_score')
sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='.4f', cbar_kws={'label': 'Mean Test Score'})
plt.title('Hyperparameter Tuning Results Heatmap')
plt.show()

heatmap_data = results.pivot_table(index='param_kernel', columns='param_degree', values='mean_test_score')
sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='.4f', cbar_kws={'label': 'Mean Test Score'})
plt.title('Hyperparameter Tuning Results Heatmap')
plt.show()

heatmap_data = results.pivot_table(index='param_kernel', columns='param_C', values='mean_test_score')
sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='.4f', cbar_kws={'label': 'Mean Test Score'})
plt.title('Hyperparameter Tuning Results Heatmap')
plt.show()