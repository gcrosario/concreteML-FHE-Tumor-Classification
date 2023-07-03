# Brain Tumor Classifier Final Training Script for Logistic Regression using Google Colaboratory

# Installing packages for concrete-ml on colab
!pip install -U pip wheel setuptools
!pip install concrete-ml


# Accessing google drive
from google.colab import drive
drive.mount('/content/gdrive')


# Importing modules
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score

from concrete.ml.sklearn import LogisticRegression
from sklearn.linear_model import LogisticRegression as skLR

import time
import pandas as pd
import numpy as np
from numpy import mean
from numpy import std


# Reading the dataset
dataset = pd.read_csv("gdrive/MyDrive/Special-Problem-Colab/Brain_GSE50161.csv")
dataset.head()

feature_cols = [c for c in dataset.columns[2:]]
x = dataset.loc[:,feature_cols].values # must be floats
y = dataset.loc[:,'type'].values # must be integers


# Preprocessing with labels for the lineage
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
le_mapping = dict(zip(le.classes_, range(len(le.classes_))))
print(le_mapping)
print(le.classes_)


# Feature Selection
from sklearn.feature_selection import SelectKBest, chi2

print("\nUsing K best features feature selection...")
print("Shape of x before selection: ", x.shape)
selector = SelectKBest(chi2, k = 20)
x_new = selector.fit_transform(x, y)
x = x_new
print("Shape of x after selection: ", x.shape)
print("\n", x)


# Get most important features accorting to Kbest
cols_index = selector.get_support(indices=True)
most_important_features = []

print("\nSelected features: ")
for col in cols_index:
  most_important_features.append(str(feature_cols[col]))
print(most_important_features)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.10)

print(f"training set size: {X_train.shape[0]}")
print(f"testing set size: {X_test.shape[0]}")


# Logistic Regression Model Training
# Scikit-learn plaintext
skmodel_LR = skLR(class_weight='balanced')
start_time = time.time()
skmodel_LR.fit(X_train,y_train)
print(f"Running time for sklearn training is {time.time() - start_time} seconds")
start_time = time.time()
y_pred_clear_LR = skmodel_LR.predict(X_test)
print(f"Running time for sklearn prediction is {time.time() - start_time} seconds")

# Quantized plaintext
quant_LR = LogisticRegression(class_weight='balanced')
start_time = time.time()
quant_LR.fit(X_train, y_train)
print(f"Running time for quantized plaintext training is {time.time() - start_time} seconds")
start_time = time.time()
y_pred_q_LR = quant_LR.predict(X_test)
print(f"Running time for quantized plaintext prediction is {time.time() - start_time} seconds")


# Metrics for scikit-learn and quantized plaintext
print("\n Logistic Regression Results \n")
# Accuracy
skLR_accuracy = accuracy_score(y_test, y_pred_clear_LR) * 100
quantLR_accuracy = accuracy_score(y_test, y_pred_q_LR) * 100
print(f"Sklearn accuracy: {skLR_accuracy:.4f}")
print(f"Quantized Clear Accuracy: {quantLR_accuracy:.4f}")
# Balanced Accuracy
skLR_bal_accuracy = balanced_accuracy_score(y_test, y_pred_clear_LR) * 100
quantLR_bal_accuracy = balanced_accuracy_score(y_test, y_pred_q_LR) * 100
print(f"Sklearn Balanced accuracy: {skLR_bal_accuracy:.4f}")
print(f"Quantized Clear Balanced Accuracy: {quantLR_bal_accuracy:.4f}")
# F1 Score
skLR_f1 = f1_score(y_test, y_pred_clear_LR, average='weighted') * 100
quantLR_f1 = f1_score(y_test, y_pred_q_LR, average='weighted') * 100
print(f"Sklearn F1 Score: {skLR_f1:.4f}")
print(f"Quantized Clear F1 Score: {quantLR_f1:.4f}")


# Logistic Regression FHE Model Compilation and Prediction on Test Set
start_time = time.time()
fhe_LR = quant_LR.compile(x)
print(f"Running time for FHE compilation is {time.time() - start_time} seconds")
start_time = time.time()
y_pred_fhe_LR = quant_LR.predict(X_test, fhe='execute')
print(f"Running time for FHE prediction is {time.time() - start_time} seconds")


# Metrics for FHE Model
print("FHE Logistic Regression Results \n")
# Accuracy
fheLR_accuracy = accuracy_score(y_test, y_pred_fhe_LR) * 100
print(f"Accuracy: {fheLR_accuracy:.4f}")
# Balanced Accuracy
fheLR_bal_accuracy = balanced_accuracy_score(y_test, y_pred_fhe_LR) * 100
print(f"Balanced accuracy: {fheLR_bal_accuracy:.4f}")
# F1 Score
fheLR_f1 = f1_score(y_test, y_pred_fhe_LR, average='weighted') * 100
print(f"F1 Score: {fheLR_f1:.4f}")


# Model prediction on test set vs Actual test set for error analysis
print("\n")
print("SKLEARN PREDICTION:\n", y_pred_clear_LR)
print("QUANTIZED CLEAR PREDICTION:\n", y_pred_q_LR)
print("FHE PREDICTION:\n", y_pred_fhe_LR)
print("ACTUAL:\n", y_test)

print("\n")
print(f"Quantized vs FHE Comparison: {int((y_pred_fhe_LR == y_pred_q_LR).sum()/len(y_pred_fhe_LR)*100)}% similar")
print(f"Sklearn vs FHE Comparison: {int((y_pred_fhe_LR == y_pred_clear_LR).sum()/len(y_pred_fhe_LR)*100)}% similar")


# Error analysis using confusion matrix
start_time = time.time()
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

print("***Note: The diagonal elements are the correctly predicted samples. ***")

print("Confusion matrix for SKLearn Plaintext: ")
sklearn_cm_display = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_clear_LR), display_labels=le.classes_)
sklearn_cm_display.plot()
plt.show()

print("Confusion matrix for Quantized Plaintext: ")
concrete_plain_display = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_q_LR), display_labels=le.classes_)
concrete_plain_display.plot()
plt.show()

print("Confusion matrix for FHE: ")
concrete_fhe_display = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_fhe_LR), display_labels=le.classes_)
concrete_fhe_display.plot()
plt.show()

print(f"Running time is {time.time() - start_time} seconds")


# Obtaining one sample per class
ependymoma_sample = dataset[dataset['samples'] == 879][most_important_features].to_numpy(dtype="uint16")
glioblastoma_sample = dataset[dataset['samples'] == 913][most_important_features].to_numpy(dtype="uint16")
medulloblastoma_sample = dataset[dataset['samples'] == 935][most_important_features].to_numpy(dtype="uint16")
normal_sample = dataset[dataset['samples'] == 948][most_important_features].to_numpy(dtype="uint16")
pilocytic_astrocytoma_sample = dataset[dataset['samples'] == 963][most_important_features].to_numpy(dtype="uint16")


# Sklearn Inference Time
average = 0

start_time = time.time()
skmodel_LR.predict(ependymoma_sample)
end_time = time.time()
print(f"Running time for Sklearn inference of ependymoma_sample is {end_time - start_time} seconds")

average += end_time - start_time

start_time = time.time()
skmodel_LR.predict(glioblastoma_sample)
end_time = time.time()
print(f"Running time for Sklearn inference of glioblastoma_sample is {end_time - start_time} seconds")

average += end_time - start_time

start_time = time.time()
skmodel_LR.predict(medulloblastoma_sample)
end_time = time.time()
print(f"Running time for Sklearn inference of medulloblastoma_sample is {end_time - start_time} seconds")

average += end_time - start_time

start_time = time.time()
skmodel_LR.predict(normal_sample)
end_time = time.time()
print(f"Running time for Sklearn inference of normal_sample is {end_time - start_time} seconds")

average += end_time - start_time

start_time = time.time()
skmodel_LR.predict(pilocytic_astrocytoma_sample)
end_time = time.time()
print(f"Running time for Sklearn inference of pilocytic_astrocytoma_sample is {end_time - start_time} seconds")

average += end_time - start_time

average /= 5

print(f"Average running time of Sklearn inference for each class is {average} seconds")


# Quantized Plaintext Inference Time
average = 0

start_time = time.time()
quant_LR.predict(ependymoma_sample)
end_time = time.time()
print(f"Running time for quantized plaintext inference of ependymoma_sample is {end_time - start_time} seconds")

average += end_time - start_time

start_time = time.time()
quant_LR.predict(glioblastoma_sample)
end_time = time.time()
print(f"Running time for quantized plaintext inference of glioblastoma_sample is {end_time - start_time} seconds")

average += end_time - start_time

start_time = time.time()
quant_LR.predict(medulloblastoma_sample)
end_time = time.time()
print(f"Running time for quantized plaintext inference of medulloblastoma_sample is {end_time - start_time} seconds")

average += end_time - start_time

start_time = time.time()
quant_LR.predict(normal_sample)
end_time = time.time()
print(f"Running time for quantized plaintext inference of normal_sample is {end_time - start_time} seconds")

average += end_time - start_time

start_time = time.time()
quant_LR.predict(pilocytic_astrocytoma_sample)
end_time = time.time()
print(f"Running time for quantized plaintext inference of pilocytic_astrocytoma_sample is {end_time - start_time} seconds")

average += end_time - start_time

average /= 5

print(f"Average running time of quantized plaintext inference for each class is {average} seconds")


# FHE Inference Time
average = 0

start_time = time.time()
quant_LR.predict(ependymoma_sample, fhe="execute")
end_time = time.time()
print(f"Running time for FHE inference of ependymoma_sample is {end_time - start_time} seconds")

average += end_time - start_time

start_time = time.time()
quant_LR.predict(glioblastoma_sample, fhe="execute")
end_time = time.time()
print(f"Running time for FHE inference of glioblastoma_sample is {end_time - start_time} seconds")

average += end_time - start_time

start_time = time.time()
quant_LR.predict(medulloblastoma_sample, fhe="execute")
end_time = time.time()
print(f"Running time for FHE inference of medulloblastoma_sample is {end_time - start_time} seconds")

average += end_time - start_time

start_time = time.time()
quant_LR.predict(normal_sample, fhe="execute")
end_time = time.time()
print(f"Running time for FHE inference of normal_sample is {end_time - start_time} seconds")

average += end_time - start_time

start_time = time.time()
quant_LR.predict(pilocytic_astrocytoma_sample, fhe="execute")
end_time = time.time()
print(f"Running time for FHE inference of pilocytic_astrocytoma_sample is {end_time - start_time} seconds")

average += end_time - start_time

average /= 5

print(f"Average running time of FHE inference for each class is {average} seconds")


# Saving the model into desired directory/path
from concrete.ml.deployment import FHEModelClient, FHEModelDev, FHEModelServer

start_time = time.time()

fhemodel_dev = FHEModelDev("gdrive/MyDrive/Special-Problem-Colab/Brain-Tumor-Models/", quant_LR)
fhemodel_dev.save()

print(f"Running time for saving the FHE model is {time.time() - start_time} seconds")


# Saving the selected features and the classes into  text file
import json

for col in cols_index:
  print(feature_cols[col])

for classLabel in le.classes_:
  print(classLabel)

with open("features_and_classes.txt", "w") as f:
    classes_list = list(le.classes_)
    temp_dict = {"features":[feature_cols[col] for col in cols_index], "classes":{classes_list.index(x):x for x in classes_list}}

    f.write(json.dumps(temp_dict))