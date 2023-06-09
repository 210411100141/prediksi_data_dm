#pip install streamlit
#pip install pandas
#pip install sklearn
#pip install plotly

# IMPORT STATEMENTS
import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv("dm.csv")

# HEADINGS
st.title('Diabetes Checkup')
# Menampilkan data
st.write(df)


tab1, tab2, = st.tabs(["Statistik Data","Tab Clustering & Silhoutte Score"])

with tab1:
   st.subheader('Training Data Stats')
   st.write(df.describe())


with tab2:
   # Fungsi untuk menampilkan grafik perbandingan diabetes vs normal pada suatu atribut
   st.subheader('Clustering Data')  
   def plot_diabetes_vs_normal(attribute):
      diabetes_data = df[df['Outcome'] == 1][attribute]
      normal_data = df[df['Outcome'] == 0][attribute]
    
      fig = plt.figure()
      plt.hist([diabetes_data, normal_data], bins=10, color=['red', 'blue'])
      plt.xlabel(attribute)
      plt.ylabel('Frequency')
      plt.legend(['Diabetes', 'Normal'])
      st.pyplot(fig)

   # Daftar atribut untuk clustering
   attributes = df.columns[:-1]  # Mengambil semua kolom kecuali kolom Outcome

   # Melakukan clustering pada setiap atribut dan menampilkan hasilnya
   for attribute in attributes:
      plot_diabetes_vs_normal(attribute)

   
   # Grafik untuk diabetes vs normal pada setiap atribut
   st.header('Grafik Diabetes vs Normal')

   # Fungsi untuk membuat grafik diabetes vs normal
   def plot_diabetes_vs_normal(attribute):
       fig = plt.figure()
       diabetes_data = df[df['Outcome'] == 1][attribute]
       normal_data = df[df['Outcome'] == 0][attribute]
       plt.hist([diabetes_data, normal_data], bins=10, color=['red', 'blue'], label=['Diabetes', 'Normal'])
       plt.xlabel(attribute)
       plt.ylabel('Frequency')
       plt.legend()
       st.pyplot(fig)

   # Melakukan plot diabetes vs normal pada setiap atribut
   for attribute in attributes:
       plot_diabetes_vs_normal(attribute)
   
    # Fungsi untuk menghitung skor siluet dari clustering K-means
   def calculate_silhouette_score(attribute):
       data = df[[attribute]].values
       scaler = MinMaxScaler()
       data_scaled = scaler.fit_transform(data)
       kmeans = KMeans(n_clusters=2, random_state=0)
       kmeans.fit(data_scaled)
       labels = kmeans.labels_
       silhouette = silhouette_score(data_scaled, labels)
       return silhouette

   # Daftar atribut untuk clustering
   attributes = df.columns[:-1]  # Mengambil semua kolom kecuali kolom Outcome

   # Melakukan perhitungan skor siluet pada setiap atribut
   silhouette_scores = {}
   for attribute in attributes:
       silhouette_scores[attribute] = calculate_silhouette_score(attribute)

   # Menampilkan skor siluet untuk setiap atribut
   st.subheader('Silhouette Score Atribut:')
   for attribute, score in silhouette_scores.items():
       st.write(f'{attribute}: {score}')

   # X AND Y DATA
   x = df.drop(['Outcome'], axis=1)
   y = df.iloc[:, -1]
   scaler = MinMaxScaler()
   x_scaled = scaler.fit_transform(x)

   # Menggabungkan semua atribut menjadi satu dataset
   all_data = df.drop('Outcome', axis=1)
   all_scaled = scaler.transform(all_data)
   kmeans_all = KMeans(n_clusters=2, random_state=0)
   kmeans_all.fit(all_scaled)
   labels_all = kmeans_all.labels_
   silhouette_all = silhouette_score(all_scaled, labels_all)

   # Menampilkan skor siluet untuk seluruh data
   st.subheader('Silhouette Score Semua Data:')
   st.write(silhouette_all)

