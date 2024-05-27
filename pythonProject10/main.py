import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import time
from sklearn.svm import SVC

# Veriyi yükleme ve ön işleme
hepC = pd.read_csv('HepatitisCdata.csv')
df = hepC.copy()
print(df.head())
print(df.info())
print(df.shape)
print(df.isnull().sum())

# Eksik değerleri sütunun ortalaması ile doldurma
df['ALB'].fillna(df['ALB'].mean(), inplace=True)
df['ALP'].fillna(df['ALP'].mean(), inplace=True)
df['CHOL'].fillna(df['CHOL'].mean(), inplace=True)
df['PROT'].fillna(df['PROT'].mean(), inplace=True)
df['ALT'].fillna(df['ALT'].mean(), inplace=True)

# Unnamed sütununu kaldırma
df = df.drop('Unnamed: 0', axis=1)

print(df.isnull().sum())

# Kategorik ve cinsiyet bilgilerini sayısal değerlere dönüştürme
df['Category'] = df['Category'].replace({'0=Blood Donor': 0, '0s=suspect Blood Donor': 0, '1=Hepatitis': 1, '2=Fibrosis': 1, '3=Cirrhosis': 1})
df['Sex'] = df['Sex'].replace({'m': 0, 'f': 1})
print(df.head())

# Korelasyon Isı Haritası oluşturma
den = df.corr()
sns.heatmap(den, annot=True, linewidths=.5, fmt=".2f")
plt.show()

# Yüksek korelasyonlu özellikleri bulma ve çıkarma
correlation_matrix = df.corr()
correlation_threshold = 0.8
correlation_mask = (np.abs(correlation_matrix) > correlation_threshold)

highly_correlated_features = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if correlation_mask.iloc[i, j]:
            feature_i = correlation_matrix.columns[i]
            feature_j = correlation_matrix.columns[j]
            highly_correlated_features.add(feature_i)
            highly_correlated_features.add(feature_j)

filtered_df = df.drop(columns=highly_correlated_features)
print(filtered_df.info())

print('Total Suspected Patients : {} '.format(df.Category.value_counts()[0]))
print('Total Healthy Patients : {} '.format(df.Category.value_counts()[1]))

# Kategorik veri dağılımı
fig, ax = plt.subplots(figsize=(8, 8))
plt.pie(x=df["Category"].value_counts(), colors=["teal", "Yellow"], labels=["Suspected Patients", "Healthy Patients"])
plt.show()

# Cinsiyet dağılımı
fig, ax = plt.subplots(figsize=(8, 8))
plt.pie(x=df["Sex"].value_counts(), colors=["Blue", "Red"], labels=["Male", "Female"], autopct="%1.2f%%")
plt.show()

print(df.info())

# Özellikler ve hedef değişkenleri ayırma
X = df.drop("Category", axis=1)
y = df["Category"]

# Veriyi train ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Modelleri tanımlama
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, solver='liblinear'),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(kernel='linear', random_state=42),
}

# Her bir model için 10 kat çapraz doğrulama ve sonuçları raporlama
for model_name, model in models.items():
    print(f"\n{model_name}")
    predictions = cross_val_predict(estimator=model, X=X, y=y, cv=10)
    report = classification_report(y, predictions)
    print(f"Classification Report for {model_name}:\n", report)

# Her bir model için eğitim ve test süreci
for model_name, model in models.items():
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    start_time = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"{model_name} Model Accuracy: {accuracy:f}")
    print(f"Classification Report for {model_name}:\n", report)
    print(f"Training Time: {train_time:.4f} seconds")
    print(f"Prediction Time: {predict_time:.4f} seconds")
    print('-' * 50)
