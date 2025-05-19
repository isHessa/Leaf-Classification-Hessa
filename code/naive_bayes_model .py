#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 20:41:39 2025

@author: hessamo
"""

# Naive Bayes Model
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# تحميل البيانات
X_train = pd.read_csv("data/preprocessed/X_train.csv")
X_test = pd.read_csv("data/preprocessed/X_test.csv")
Y_train = pd.read_csv("data/preprocessed/Y_train.csv").values.ravel()
Y_test = pd.read_csv("data/preprocessed/Y_test.csv").values.ravel()

# إنشاء النموذج وتدريبه
model = GaussianNB()
model.fit(X_train, Y_train)

# التنبؤ
predictions = model.predict(X_test)

# دقة النموذج
accuracy = accuracy_score(Y_test, predictions)
print("🎯 دقة النموذج Naive Bayes:", accuracy)

# حفظ التنبؤات
pd.DataFrame(predictions, columns=["Predicted"]).to_csv("data/Results/predctions_NaiveBayes_model.csv", index=False)