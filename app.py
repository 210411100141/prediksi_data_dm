import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset diabetes
diabetes = load_diabetes('dm.csv')
X = diabetes.data
y = diabetes.target

# Membagi dataset menjadi data training dan data testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Menerapkan metode k-means
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train)

# Mendapatkan cluster label untuk data training
train_labels = kmeans.labels_

# Mendapatkan cluster label untuk data testing
test_labels = kmeans.predict(X_test)

# Menghitung akurasi
accuracy = accuracy_score(y_test, test_labels)

# Menampilkan grafik
plt.scatter(X_train[:, 0], X_train[:, 1], c=train_labels)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='*', s=200, c='red')
plt.title('Prediksi Penyakit Diabetes menggunakan K-means')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Menampilkan akurasi dan nilai output
print("Akurasi prediksi: {:.2f}%".format(accuracy * 100))
print("Cluster label untuk data testing:")
print(test_labels)
