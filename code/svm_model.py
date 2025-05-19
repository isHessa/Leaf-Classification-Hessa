#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 20:38:18 2025

@author: hessamo
"""
# Support Vector Machine Model
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# تحميل البيانات
X_train = pd.read_csv("data/preprocessed/X_train.csv")
X_test = pd.read_csv("data/preprocessed/X_test.csv")
Y_train = pd.read_csv("data/preprocessed/Y_train.csv").values.ravel()
Y_test = pd.read_csv("data/preprocessed/Y_test.csv").values.ravel()

# إنشاء النموذج وتدريبه
model = SVC(kernel='linear', random_state=42)
model.fit(X_train, Y_train)

# التنبؤ
predictions = model.predict(X_test)

# دقة النموذج
accuracy = accuracy_score(Y_test, predictions)
print("🎯 دقة النموذج SVM:", accuracy)

# حفظ التنبؤات
pd.DataFrame(predictions, columns=["Predicted"]).to_csv("data/Results/predctions_SVM_model.csv", index=False)
