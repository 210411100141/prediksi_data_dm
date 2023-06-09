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
st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')
st.write(df.describe())

tab1, tab2, = st.tabs(["Deskripsi Data","Tab Visualisasi data"])

with tab1:
   # Fungsi untuk menampilkan grafik perbandingan diabetes vs normal pada suatu atribut
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

with tab2:
   # X AND Y DATA
   x = df.drop(['Outcome'], axis=1)
   y = df.iloc[:, -1]
   scaler = MinMaxScaler()
   x_scaled = scaler.fit_transform(x)
   
   # FUNCTION
   def user_report():
      usia = st.sidebar.slider('Usia', 15, 80, 20)
      glukosa = st.sidebar.slider('Glukosa', 70, 200, 90)
      td = st.sidebar.slider('Tekanan darah', 0, 110, 70)
      kk = st.sidebar.slider('Ketebalan Kulit', 0, 50, 25)
      insulin = st.sidebar.slider('Insulin', 0, 900, 200)
      bmi = st.sidebar.slider('BMI', 20, 450, 50)
      
      
      user_report_data = {
         'usia': usia,
         'glukosa': glukosa,
         'td': td,
         'kk': kk,
         'insulin': insulin,
         'bmi': bmi,
      }
      report_data = pd.DataFrame(user_report_data, index=[0])
      return report_data
     
   # PATIENT DATA
   user_data = user_report()
   st.subheader('Patient Data')
   st.write(user_data)
    
   # MODEL
   kmeans = KMeans(n_clusters=2, random_state=0)
   kmeans.fit(x_scaled)
   user_result = kmeans.predict(scaler.transform(user_data))
   cluster_labels = kmeans.labels_
  
   # MODEL SILUET
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

st.subheader('Silhouette Scores:')
for attribute, score in silhouette_scores.items():
    st.write(f'{attribute}: {score}')
      
   # COLOR FUNCTION
   if user_result[0] == 0:
      color = 'blue'
   else:
      color = 'red'
   
   # usia vs glukosa
   st.header('Glukosa count Graph (Others vs Yours)')
   fig_glukosa = plt.figure()
   plt.scatter(df['Usia'], df['Glukosa'], c=kmeans.labels_, cmap='viridis')
   plt.scatter(user_data['usia'], user_data['glukosa'], s=150, color=color)
   plt.xticks(np.arange(10, 100, 10))
   plt.yticks(np.arange(60, 200, 10))
   plt.title('0 - Cluster 1 & 1 - Cluster 2')
   st.pyplot(fig_glukosa)
   
   # usia vs tekanan darah
   st.header('Tekanan Darah Graph (Others vs Yours)')
   fig_td = plt.figure()
   plt.scatter(df['Usia'], df['Tekanan darah'], c=kmeans.labels_, cmap='viridis')
   plt.scatter(user_data['usia'], user_data['td'], s=150, color=color)
   plt.xticks(np.arange(10, 100, 10))
   plt.yticks(np.arange(0, 120, 10))
   plt.title('0 - Cluster 1 & 1 - Cluster 2')
   st.pyplot(fig_td)

   # usia vs ketebalan kulit
   st.header('Ketebalan Kulit Value Graph (Others vs Yours)')
   fig_kk = plt.figure()
   plt.scatter(df['Usia'], df['Ketebalan Kulit'], c=kmeans.labels_, cmap='viridis')
   plt.scatter(user_data['usia'], user_data['kk'], s=150, color=color)
   plt.xticks(np.arange(10, 100, 10))
   plt.yticks(np.arange(0, 60, 5))
   plt.title('0 - Cluster 1 & 1 - Cluster 2')
   st.pyplot(fig_kk)

   # usia vs insulin
   st.header('Insulin Value Graph (Others vs Yours)')
   fig_insulin = plt.figure()
   plt.scatter(df['Usia'], df['Insulin'], c=kmeans.labels_, cmap='viridis')
   plt.scatter(user_data['usia'], user_data['insulin'], s=150, color=color)
   plt.xticks(np.arange(10, 100, 10))
   plt.yticks(np.arange(0, 920, 50))
   plt.title('0 - Cluster 1 & 1 - Cluster 2')
   st.pyplot(fig_insulin)

   # Age vs BMI
   st.header('BMI Value Graph (Others vs Yours)')
   fig_bmi = plt.figure()
   plt.scatter(df['Usia'], df['BMI'], c=kmeans.labels_, cmap='viridis')
   plt.scatter(user_data['usia'], user_data['bmi'], s=150, color=color)
   plt.xticks(np.arange(10, 100, 10))
   plt.yticks(np.arange(10, 470, 25))
   plt.title('0 - Cluster 1 & 1 - Cluster 2')
   st.pyplot(fig_bmi)

# OUTPUT
st.subheader('Your Report: ')
output = ''
if user_result[0] == 0:
  output = 'You belong to Cluster 1'
else:
  output = 'You belong to Cluster 2'
st.title(output)
st.subheader('Silhouette Scores:')
for attribute, score in silhouette_scores.items():
    st.write(f'{attribute}: {score}')

