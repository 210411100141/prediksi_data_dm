import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def kmeans_clustering(data, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(data)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    return labels, centroids

def main():
    st.title("Aplikasi K-means Clustering")

    # Upload dataset
    st.subheader("Upload Dataset")
    file = st.file_uploader("Upload file CSV", type=["csv"])
    if file is not None:
        data = pd.read_csv(dm.scv)

        # Menampilkan data
        st.subheader("Data")
        st.write(data)

        # Memilih atribut yang digunakan untuk clustering
        selected_features = st.multiselect("Pilih fitur untuk clustering", data.columns)

        # Memilih jumlah cluster
        num_clusters = st.slider("Pilih jumlah cluster", min_value=2, max_value=10)

        if st.button("Clustering"):
            selected_data = data[selected_features]
            labels, centroids = kmeans_clustering(selected_data, num_clusters)

            # Menambahkan kolom hasil clustering pada dataset
            data["Cluster"] = labels

            # Menampilkan hasil clustering
            st.subheader("Hasil Clustering")
            st.write(data)

            # Menampilkan nilai centroid untuk setiap cluster
            st.subheader("Nilai Centroid")
            st.write(pd.DataFrame(centroids, columns=selected_features))

if __name__ == "__main__":
    main()
