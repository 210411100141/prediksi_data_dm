#pip install streamlit
#pip install pandas
#pip install sklearn


# IMPORT STATEMENTS
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("diabetes.csv")

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
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x)
    
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
        report_data_scaled = scaler.transform(report_data)
        return report_data_scaled

    # PATIENT DATA
    user_data = user_report()
    st.subheader('Patient Data')
    st.write(user_data)

    # MODEL
    k = 2  # Number of clusters
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(x_train)
    user_result = kmeans.predict(user_data)

    # COLOR FUNCTION
    if user_result[0] == 0:
        color = 'blue'
    else:
        color = 'red'

    # Age vs Glucose
    st.header('Glucose Value Graph (Others vs Yours)')
    fig_glucose = plt.figure()
    ax3 = sns.scatterplot(x='Age', y='Glucose', data=df, hue='Outcome', palette='magma')
    ax4 = sns.scatterplot(x=user_data[:, 7], y=user_data[:, 1], s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 220, 10))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_glucose)

    # Age vs Bp
    st.header('Blood Pressure Value Graph (Others vs Yours)')
    fig_bp = plt.figure()
    ax5 = sns.scatterplot(x='Age', y='BloodPressure', data=df, hue='Outcome', palette='Reds')
    ax6 = sns.scatterplot(x=user_data[:, 7], y=user_data[:, 2], s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 130, 10))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_bp)

# OUTPUT
st.subheader('Your Report: ')
output = ''
if user_result[0] == 0:
    output = 'You are not Diabetic'
else:
    output = 'You are Diabetic'
st.title(output)
st.subheader('Silhouette Score: ')
st.write(str(silhouette_score(x_train, kmeans.labels_)) + '%')
