import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Fungsi untuk menghitung akurasi
def calculate_accuracy(true_labels, predicted_labels):
    return accuracy_score(true_labels, predicted_labels)

# Membaca data dari file CSV
data = pd.read_csv('dm.csv')

# Mengubah data menjadi array numpy
X = np.array(data.drop('class', axis=1))

# Menginisialisasi model K-Means
kmeans = KMeans(n_clusters=2, random_state=0)

# Melatih model K-Means
kmeans.fit(X)

# Mendapatkan label prediksi
predicted_labels = kmeans.labels_

# Mendapatkan nilai centroid
centroids = kmeans.cluster_centers_

# Menambahkan kolom 'predicted_class' ke dalam DataFrame
data['predicted_class'] = predicted_labels

# Menampilkan grafik hasil klasterisasi
plt.scatter(X[:, 0], X[:, 1], c=predicted_labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Menghitung akurasi
true_labels = np.array(data['class'])
accuracy = calculate_accuracy(true_labels, predicted_labels)
print("Akurasi: {:.2f}%".format(accuracy * 100))

# Menampilkan nilai output
print("Nilai Output:")
print(data)
