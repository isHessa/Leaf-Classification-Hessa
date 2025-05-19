#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 20:45:46 2025

@author: hessamo
"""

# Artificial Neural Network Model (ANN)
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# تحميل البيانات
X_train = pd.read_csv("data/preprocessed/X_train.csv")
X_test = pd.read_csv("data/preprocessed/X_test.csv")
Y_train = pd.read_csv("data/preprocessed/Y_train.csv").values.ravel()
Y_test = pd.read_csv("data/preprocessed/Y_test.csv").values.ravel()

# إنشاء النموذج وتدريبه
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
model.fit(X_train, Y_train)

# التنبؤ
predictions = model.predict(X_test)

# دقة النموذج
accuracy = accuracy_score(Y_test, predictions)
print("🤖 دقة النموذج ANN:", accuracy)

# حفظ التنبؤات
pd.DataFrame(predictions, columns=["Predicted"]).to_csv("data/Results/predctions_ANN_model.csv", index=False)