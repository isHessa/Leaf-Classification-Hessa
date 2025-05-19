#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 20:22:21 2025

@author: hessamo
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os

# إنشاء المجلدات إذا ما كانت موجودة
os.makedirs("data/preprocessed", exist_ok=True)
os.makedirs("data/Results", exist_ok=True)

# تحميل البيانات الأصلية
df = pd.read_csv("data/original/train.csv")

# عرض أول 5 صفوف للتأكد
print("👀 معاينة أول 5 صفوف من البيانات:")
print(df.head())

# فصل الخصائص (X) عن المخرجات (Y)
X = df.drop(columns=["id", "species"])
Y = df["species"]

# حفظ البيانات الكاملة (قبل التقسيم)
X.to_csv("data/preprocessed/X.csv", index=False)
Y.to_csv("data/preprocessed/Y.csv", index=False)

# تقسيم البيانات إلى تدريب واختبار 80/20
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# حفظ ملفات التدريب والاختبار
X_train.to_csv("data/preprocessed/X_train.csv", index=False)
Y_train.to_csv("data/preprocessed/Y_train.csv", index=False)
X_test.to_csv("data/preprocessed/X_test.csv", index=False)
Y_test.to_csv("data/preprocessed/Y_test.csv", index=False)

print("✅ تم تجهيز البيانات وتخزين الملفات في مجلد preprocessed.")