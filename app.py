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
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv("dm.csv")

# HEADINGS
st.title('Diabetes Checkup')
st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')
st.write(df.describe())

tab1, tab2 = st.tabs(["Deskripsi Data", "Tab Visualisasi data"])

with tab1:
   st.image("ar.png")

with tab2:
   # X AND Y DATA
   x = df.drop(['Outcome'], axis=1)
   y = df.iloc[:, -1]
   scaler = MinMaxScaler()
   x_scaled = scaler.fit_transform(x)
   
   # FUNCTION
   def user_report():
      pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
      glucose = st.sidebar.slider('Glucose', 0, 200, 120)
      bp = st.sidebar.slider('Blood Pressure', 0, 122, 70)
      skinthickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
      insulin = st.sidebar.slider('Insulin', 0, 846, 79)
      bmi = st.sidebar.slider('BMI', 0, 67, 20)
      dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
      age = st.sidebar.slider('Age', 21, 88, 33)
      
      user_report_data = {
         'pregnancies': pregnancies,
         'glucose': glucose,
         'bp': bp,
         'skinthickness': skinthickness,
         'insulin': insulin,
         'bmi': bmi,
         'dpf': dpf,
         'age': age
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
      
   # COLOR FUNCTION
   if user_result[0] == 0:
      color = 'blue'
   else:
      color = 'red'
   
   # Age vs Pregnancies
   st.header('Pregnancy count Graph (Others vs Yours)')
   fig_preg = plt.figure()
   plt.scatter(df['Age'], df['Pregnancies'], c=kmeans.labels_, cmap='viridis')
   plt.scatter(user_data['age'], user_data['pregnancies'], s=150, color=color)
   plt.xticks(np.arange(10, 100, 5))
   plt.yticks(np.arange(0, 20, 2))
   plt.title('0 - Cluster 1 & 1 - Cluster 2')
   st.pyplot(fig_preg)
   
   # Age vs Glucose
   st.header('Glucose Value Graph (Others vs Yours)')
   fig_glucose = plt.figure()
   plt.scatter(df['Age'], df['Glucose'], c=kmeans.labels_, cmap='viridis')
   plt.scatter(user_data['age'], user_data['glucose'], s=150, color=color)
   plt.xticks(np.arange(10, 100, 5))
   plt.yticks(np.arange(0, 220, 10))
   plt.title('0 - Cluster 1 & 1 - Cluster 2')
   st.pyplot(fig_glucose)

   # Age vs Bp
   st.header('Blood Pressure Value Graph (Others vs Yours)')
   fig_bp = plt.figure()
   plt.scatter(df['Age'], df['BloodPressure'], c=kmeans.labels_, cmap='viridis')
   plt.scatter(user_data['age'], user_data['bp'], s=150, color=color)
   plt.xticks(np.arange(10, 100, 5))
   plt.yticks(np.arange(0, 130, 10))
   plt.title('0 - Cluster 1 & 1 - Cluster 2')
   st.pyplot(fig_bp)

   # Age vs St
   st.header('Skin Thickness Value Graph (Others vs Yours)')
   fig_st = plt.figure()
   plt.scatter(df['Age'], df['SkinThickness'], c=kmeans.labels_, cmap='viridis')
   plt.scatter(user_data['age'], user_data['skinthickness'], s=150, color=color)
   plt.xticks(np.arange(10, 100, 5))
   plt.yticks(np.arange(0, 110, 10))
   plt.title('0 - Cluster 1 & 1 - Cluster 2')
   st.pyplot(fig_st)

   # Age vs Insulin
   st.header('Insulin Value Graph (Others vs Yours)')
   fig_i = plt.figure()
   plt.scatter(df['Age'], df['Insulin'], c=kmeans.labels_, cmap='viridis')
   plt.scatter(user_data['age'], user_data['insulin'], s=150, color=color)
   plt.xticks(np.arange(10, 100, 5))
   plt.yticks(np.arange(0, 900, 50))
   plt.title('0 - Cluster 1 & 1 - Cluster 2')
   st.pyplot(fig_i)

   # Age vs BMI
   st.header('BMI Value Graph (Others vs Yours)')
   fig_bmi = plt.figure()
   plt.scatter(df['Age'], df['BMI'], c=kmeans.labels_, cmap='viridis')
   plt.scatter(user_data['age'], user_data['bmi'], s=150, color=color)
   plt.xticks(np.arange(10, 100, 5))
   plt.yticks(np.arange(0, 70, 5))
   plt.title('0 - Cluster 1 & 1 - Cluster 2')
   st.pyplot(fig_bmi)

   # Age vs Dpf
   st.header('DPF Value Graph (Others vs Yours)')
   fig_dpf = plt.figure()
   plt.scatter(df['Age'], df['DiabetesPedigreeFunction'], c=kmeans.labels_, cmap='viridis')
   plt.scatter(user_data['age'], user_data['dpf'], s=150, color=color)
   plt.xticks(np.arange(10, 100, 5))
   plt.yticks(np.arange(0, 3, 0.2))
   plt.title('0 - Cluster 1 & 1 - Cluster 2')
   st.pyplot(fig_dpf)

# OUTPUT
st.subheader('Your Report: ')
output = ''
if user_result[0] == 0:
  output = 'You belong to Cluster 1'
else:
  output = 'You belong to Cluster 2'
st.title(output)
st.subheader('Accuracy: ')
st.write('N/A')

