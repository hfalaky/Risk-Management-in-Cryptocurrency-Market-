from sklearn.datasets import load_iris
import pandas as pd

# Load the Iris dataset
iris = load_iris()
# Convert to DataFrame
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)                  #to convert from array to dataset
iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)          #converts numeric class labels into names
# Display the first few rows of the dataset
iris_df.head()

#print(iris_df.head())

from ydata_profiling import ProfileReport           # automatically analyzes a dataset and produces stats like profile like last tp 
# Generate the profiling report
profile = ProfileReport(iris_df, title="Iris Dataset Profiling Report")
#profile.to_notebook_iframe()
profile.to_file('eeeee.html')               #repalced with lab1 line



# the correlation matrix shows that petal length and petal width are very much correlated with each other and with the species label
#  sepal length has a moderate correlation, while sepal width shows weak and negative correlations

# plot:
import seaborn as sns
import matplotlib.pyplot as plt

corr_matrix = iris_df.drop(columns=["species"]).corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.tight_layout()
plt.savefig("correlation_matrix.png", dpi=200)
plt.close()
print("matrix saved as png")


# ------- SECTION 3 ------

# Create a copy and add is_virginica flag
virginica_df = iris_df.copy()
virginica_df["is_virginica"] = (iris.target == 2).astype(int)       #create column flag
# Separate features and label
X = virginica_df[iris.feature_names]        #includes all features, predefined (used fo2)
y = virginica_df["is_virginica"]        #one vs all
# Display rows
#print(virginica_df.head(150))

#we want to check how many in is viriginca=1:
print(y.value_counts())             #so 50 rows =1 which means 50 are virginica and 100 are not


# converting the target variable to 0 or 1 is useful because logistic regression models the probability of a binary outcome
#  using 0 and 1 allows the model to interpret the target as the probability of belonging to the positive class 1



#-------taining and testing split-------

# Train_split with stratification
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42,          #random state is same idea as seed, makes it reproducible/accessible again
stratify=y) 
#stratify=y ensures that the training and test sets maintain the same class proportions as the original dataset, preventing class imbalance after splitting.
print(y_train.value_counts(), y_test.value_counts())

# originally it was 100 / 50
# for training data 70% :   70 / 35
# and for testing data:    30 / 15 


# -------feature scaling -------

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)



#------- Binary logistic regression on standardized features-------

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(max_iter=500, solver="lbfgs")       #max itteration is for gradient decent?
logreg.fit(X_train_std, y_train)        #training the data 
print("Coefficients (per feature):", dict(zip(iris.feature_names, logreg.coef_[0])))
print("Intercept:", logreg.intercept_[0])

# Larger petal length and petal width strongly favor the Virginica class 
# alors que larger sepal width slightly reduces the likelihood of Virginica (negative)

# The most predictive feature is the one with the largest absolute coefficient: (petal width: 2.74)



# ------- Predict labels and probabilities  -------

# Import necessary libraries

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Use the trained logistic regression model (logreg) to make predictions on the standardized test set
y_pred = logreg.predict(X_test_std)
# Compute the predicted probabilities for the positive class (Virginica = 1)
# predict_proba returns probabilities for all classes; [:, 1] selects the column for class 1
y_proba = logreg.predict_proba(X_test_std)[:, 1]
# Evaluate model performance
# Print the overall accuracy of the model
print("Accuracy:", accuracy_score(y_test, y_pred))
# Print a detailed classification report (precision, recall, f1-score, support) for each class
print(classification_report(y_test, y_pred, target_names=["Not Virginica", "Virginica"]))
# Print the confusion matrix to see the distribution of correct and incorrect predictions
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))


# 3 false negs

#recall for Virginica is more important than precision when the goal is to identify all Virginica flowers and missing one is costly

#------- Threshold tuning-------

# Import the function to compute precision, recall, F1-score, and support
from sklearn.metrics import precision_recall_fscore_support
# Define a helper function to evaluate performance at a specific probability threshold
from sklearn.metrics import precision_recall_fscore_support

# Define a helper function to evaluate performance at a specific probability threshold
def evaluate_at(th):
    # Convert predicted probabilities into binary predictions
    y_hat = (y_proba >= th).astype(int)

    # Compute precision, recall, and F1-score for the positive class (Virginica)
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, y_hat, average="binary"
    )

    return th, p, r, f1

# Evaluate the model at different probability thresholds
for th in [0.3, 0.5, 0.7]:
    th, p, r, f1 = evaluate_at(th)
    print(f"Threshold={th:.2f}  Precision={p:.3f}  Recall={r:.3f}  F1={f1:.3f}")


#High recall is critical when missing a Virginica flower (false negative) is costly.
#A threshold of 0.30 is preferred because it maximizes recall for Virginica while maintaining 
# a good balance between precision and recall -->suitable for recall/ critical use cases.
