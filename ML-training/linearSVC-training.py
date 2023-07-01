# Brain Tumor Classifier Final Training Script for Linear SVC using Google Colaboratory

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

from concrete.ml.sklearn.svm import LinearSVC
from sklearn.svm import LinearSVC as skSVC

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


# Linear SVC Model Training
# Scikit-learn plaintext
skmodel_SVC = skSVC(class_weight='balanced')
start_time = time.time()
skmodel_SVC.fit(X_train,y_train)
print(f"Running time for sklearn training is {time.time() - start_time} seconds")
start_time = time.time()
y_pred_clear_SVC = skmodel_SVC.predict(X_test)
print(f"Running time for sklearn prediction is {time.time() - start_time} seconds")

# Quantized plaintext
quant_SVC = LinearSVC(class_weight='balanced')
start_time = time.time()
quant_SVC.fit(X_train, y_train)
print(f"Running time for quantized plaintext training is {time.time() - start_time} seconds")
start_time = time.time()
y_pred_q_SVC = quant_SVC.predict(X_test)
print(f"Running time for quantized plaintext prediction is {time.time() - start_time} seconds")


# Metrics for scikit-learn and quantized plaintext
print("\n Linear SVC Results \n")
# Accuracy
skSVC_accuracy = accuracy_score(y_test, y_pred_clear_SVC) * 100
quantSVC_accuracy = accuracy_score(y_test, y_pred_q_SVC) * 100
print(f"Sklearn accuracy: {skSVC_accuracy:.4f}")
print(f"Quantized Clear Accuracy: {quantSVC_accuracy:.4f}")
# Balanced Accuracy
skSVC_bal_accuracy = balanced_accuracy_score(y_test, y_pred_clear_SVC) * 100
quantSVC_bal_accuracy = balanced_accuracy_score(y_test, y_pred_q_SVC) * 100
print(f"Sklearn Balanced accuracy: {skSVC_bal_accuracy:.4f}")
print(f"Quantized Clear Balanced Accuracy: {quantSVC_bal_accuracy:.4f}")
# F1 Score
skSVC_f1 = f1_score(y_test, y_pred_clear_SVC, average='weighted') * 100
quantSVC_f1 = f1_score(y_test, y_pred_q_SVC, average='weighted') * 100
print(f"Sklearn F1 Score: {skSVC_f1:.4f}")
print(f"Quantized Clear F1 Score: {quantSVC_f1:.4f}")


# Linear SVC FHE Model Compilation and Prediction on Test Set
start_time = time.time()
fhe_SVC = quant_SVC.compile(x)
print(f"Running time for FHE compilation is {time.time() - start_time} seconds")
start_time = time.time()
y_pred_fhe_SVC = quant_SVC.predict(X_test, fhe="execute")
print(f"Running time for FHE prediction is {time.time() - start_time} seconds")


# Metrics for FHE Model
print("FHE Linear SVC Results \n")
# Accuracy
fheSVC_accuracy = accuracy_score(y_test, y_pred_fhe_SVC) * 100
print(f"Accuracy: {fheSVC_accuracy:.4f}")
# Balanced Accuracy
fheSVC_bal_accuracy = balanced_accuracy_score(y_test, y_pred_fhe_SVC) * 100
print(f"Balanced accuracy: {fheSVC_bal_accuracy:.4f}")
# F1 Score
fheRF_f1 = f1_score(y_test, y_pred_fhe_SVC, average='weighted') * 100
print(f"F1 Score: {fheRF_f1:.4f}")


# Model prediction on test set vs Actual test set
print("\n")
print("SKLEARN PREDICTION:\n", y_pred_clear_SVC)
print("QUANTIZED CLEAR PREDICTION:\n", y_pred_q_SVC)
print("FHE PREDICTION:\n", y_pred_fhe_SVC)
print("ACTUAL:\n", y_test)

print("\n")
print(f"Quantized vs FHE Comparison: {int((y_pred_fhe_SVC == y_pred_q_SVC).sum()/len(y_pred_fhe_SVC)*100)}% similar")
print(f"Sklearn vs FHE Comparison: {int((y_pred_fhe_SVC == y_pred_clear_SVC).sum()/len(y_pred_fhe_SVC)*100)}% similar")


# Obtaining one sample per class
ependymoma_sample = dataset[dataset['samples'] == 879][most_important_features].to_numpy(dtype="uint16")
glioblastoma_sample = dataset[dataset['samples'] == 913][most_important_features].to_numpy(dtype="uint16")
medulloblastoma_sample = dataset[dataset['samples'] == 935][most_important_features].to_numpy(dtype="uint16")
normal_sample = dataset[dataset['samples'] == 948][most_important_features].to_numpy(dtype="uint16")
pilocytic_astrocytoma_sample = dataset[dataset['samples'] == 963][most_important_features].to_numpy(dtype="uint16")


# Sklearn Inference Time
average = 0

start_time = time.time()
skmodel_SVC.predict(ependymoma_sample)
end_time = time.time()
print(f"Running time for Sklearn inference of ependymoma_sample is {end_time - start_time} seconds")

average += end_time - start_time

start_time = time.time()
skmodel_SVC.predict(glioblastoma_sample)
end_time = time.time()
print(f"Running time for Sklearn inference of glioblastoma_sample is {end_time - start_time} seconds")

average += end_time - start_time

start_time = time.time()
skmodel_SVC.predict(medulloblastoma_sample)
end_time = time.time()
print(f"Running time for Sklearn inference of medulloblastoma_sample is {end_time - start_time} seconds")

average += end_time - start_time

start_time = time.time()
skmodel_SVC.predict(normal_sample)
end_time = time.time()
print(f"Running time for Sklearn inference of normal_sample is {end_time - start_time} seconds")

average += end_time - start_time

start_time = time.time()
skmodel_SVC.predict(pilocytic_astrocytoma_sample)
end_time = time.time()
print(f"Running time for Sklearn inference of pilocytic_astrocytoma_sample is {end_time - start_time} seconds")

average += end_time - start_time

average /= 5

print(f"Average running time of Sklearn inference for each class is {average} seconds")


# Quantized Plaintext Inference Time
average = 0

start_time = time.time()
quant_SVC.predict(ependymoma_sample)
end_time = time.time()
print(f"Running time for quantized plaintext inference of ependymoma_sample is {end_time - start_time} seconds")

average += end_time - start_time

start_time = time.time()
quant_SVC.predict(glioblastoma_sample)
end_time = time.time()
print(f"Running time for quantized plaintext inference of glioblastoma_sample is {end_time - start_time} seconds")

average += end_time - start_time

start_time = time.time()
quant_SVC.predict(medulloblastoma_sample)
end_time = time.time()
print(f"Running time for quantized plaintext inference of medulloblastoma_sample is {end_time - start_time} seconds")

average += end_time - start_time

start_time = time.time()
quant_SVC.predict(normal_sample)
end_time = time.time()
print(f"Running time for quantized plaintext inference of normal_sample is {end_time - start_time} seconds")

average += end_time - start_time

start_time = time.time()
quant_SVC.predict(pilocytic_astrocytoma_sample)
end_time = time.time()
print(f"Running time for quantized plaintext inference of pilocytic_astrocytoma_sample is {end_time - start_time} seconds")

average += end_time - start_time

average /= 5

print(f"Average running time of quantized plaintext inference for each class is {average} seconds")


# FHE Inference Time
average = 0

start_time = time.time()
quant_SVC.predict(ependymoma_sample, fhe="execute")
end_time = time.time()
print(f"Running time for FHE inference of ependymoma_sample is {end_time - start_time} seconds")

average += end_time - start_time

start_time = time.time()
quant_SVC.predict(glioblastoma_sample, fhe="execute")
end_time = time.time()
print(f"Running time for FHE inference of glioblastoma_sample is {end_time - start_time} seconds")

average += end_time - start_time

start_time = time.time()
quant_SVC.predict(medulloblastoma_sample, fhe="execute")
end_time = time.time()
print(f"Running time for FHE inference of medulloblastoma_sample is {end_time - start_time} seconds")

average += end_time - start_time

start_time = time.time()
quant_SVC.predict(normal_sample, fhe="execute")
end_time = time.time()
print(f"Running time for FHE inference of normal_sample is {end_time - start_time} seconds")

average += end_time - start_time

start_time = time.time()
quant_SVC.predict(pilocytic_astrocytoma_sample, fhe="execute")
end_time = time.time()
print(f"Running time for FHE inference of pilocytic_astrocytoma_sample is {end_time - start_time} seconds")

average += end_time - start_time

average /= 5

print(f"Average running time of FHE inference for each class is {average} seconds")


# Saving the model into desired directory/path
from concrete.ml.deployment import FHEModelClient, FHEModelDev, FHEModelServer

start_time = time.time()

fhemodel_dev = FHEModelDev("gdrive/MyDrive/Special-Problem-Colab/Brain-Tumor-Models/", quant_SVC)
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