import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# Mengambil data diabetes dari file CSV atau DataFrame Anda
data_path = "dm.csv"
df = pd.read_csv(data_path)

# Menghilangkan kolom Outcome karena itu adalah label yang akan diprediksi
X = df.drop("Outcome", axis=1)

# Membuat fungsi untuk melakukan clustering dengan metode K-means
def kmeans_clustering(data, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(data)
    labels = kmeans.labels_
    return labels

# Membuat halaman dengan Streamlit
def main():
    st.title("Aplikasi Clustering K-means untuk Data Penyakit Diabetes")
    
    # Menampilkan data awal
    st.subheader("Data Penyakit Diabetes")
    st.dataframe(df)
    
    # Mengambil jumlah cluster dari pengguna
    num_clusters = st.slider("Jumlah Cluster", min_value=2, max_value=10, value=3)
    
    # Melakukan clustering dengan metode K-means
    labels = kmeans_clustering(X, num_clusters)
    
    # Menampilkan hasil clustering
    st.subheader("Hasil Clustering")
    df_result = df.copy()
    df_result["Cluster"] = labels
    st.dataframe(df_result)
    
    # Menghitung akurasi clustering dengan menggunakan kolom "Outcome" sebagai label
    true_labels = df["Outcome"]
    accuracy = accuracy_score(true_labels, labels)
    
    # Menampilkan akurasi clustering
    st.subheader("Akurasi Clustering")
    st.write(f"Akurasi: {accuracy}")
    
if __name__ == "__main__":
    main()
