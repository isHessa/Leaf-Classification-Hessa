#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 20:40:24 2025

@author: hessamo
"""

# K-Nearest Neighbors Model
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# تحميل البيانات
X_train = pd.read_csv("data/preprocessed/X_train.csv")
X_test = pd.read_csv("data/preprocessed/X_test.csv")
Y_train = pd.read_csv("data/preprocessed/Y_train.csv").values.ravel()
Y_test = pd.read_csv("data/preprocessed/Y_test.csv").values.ravel()

# إنشاء النموذج وتدريبه
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, Y_train)

# التنبؤ
predictions = model.predict(X_test)

# دقة النموذج
accuracy = accuracy_score(Y_test, predictions)
print("🎯 دقة النموذج KNN:", accuracy)

# حفظ التنبؤات
pd.DataFrame(predictions, columns=["Predicted"]).to_csv("data/Results/predctions_KNN_model.csv", index=False)