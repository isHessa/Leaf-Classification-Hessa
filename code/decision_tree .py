#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 20:34:01 2025

@author: hessamo
"""

# Decision Tree Classifier Model
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# تحميل البيانات المعالجة
X_train = pd.read_csv("data/preprocessed/X_train.csv")
X_test = pd.read_csv("data/preprocessed/X_test.csv")
Y_train = pd.read_csv("data/preprocessed/Y_train.csv").values.ravel()
Y_test = pd.read_csv("data/preprocessed/Y_test.csv").values.ravel()

# إنشاء النموذج وتدريبه
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, Y_train)

# إجراء التنبؤات
predictions = model.predict(X_test)

# حساب الدقة
accuracy = accuracy_score(Y_test, predictions)
print("📊 دقة النموذج:", accuracy)

# حفظ التنبؤات في ملف CSV
pd.DataFrame(predictions, columns=["Predicted"]).to_csv("data/Results/predictions_DecisionTree.csv", index=False)