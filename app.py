#pip install streamlit
#pip install pandas
#pip install sklearn
#pip install plotly

# IMPORT STATEMENTS
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import seaborn as sns

df = pd.read_csv("dm.csv")

# HEADINGS
st.title('Prediksi Diabetes')
st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')
st.write(df.describe())

tab1, tab2 = st.tabs(["Deskripsi Data", "Tab Visualisasi data"])

with tab1:
    st.image("ar.png")

with tab2:
    # X AND Y DATA
    x = df.drop(['Outcome'], axis = 1)
    y = df.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state=0)

    # FUNCTION
    def user_report():
        usia = st.sidebar.slider('Usia', 0, 17, 3)
        glukosa = st.sidebar.slider('Glukosa', 0, 200, 120)
        td = st.sidebar.slider('Tekanan Darah', 0, 122, 70)
        kk = st.sidebar.slider('Ketebalan Kulit', 0, 100, 20)
        insulin = st.sidebar.slider('Insulin', 0, 846, 79)
        bmi = st.sidebar.slider('BMI', 0, 67, 20)

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
    kmeans.fit(x_train, y_train)
    user_result = kmeans.predict(user_data)

    # COLOR FUNCTION
    if user_result[0] == 0:
        color = 'blue'
    else:
        color = 'red'

    # Usia vs Glukosa
    st.header('Pregnancy count Graph (Others vs Yours)')
    fig_preg = plt.figure()
    ax1 = sns.scatterplot(x='Usia', y='Glukosa', data=df, hue='Outcome', palette='Greens')
    ax2 = sns.scatterplot(x=user_data['usia'], y=user_data['glukosa'], s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 20, 2))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_glu)

    # Usia vs Tekanan Darah
    st.header('Glucose Value Graph (Others vs Yours)')
    fig_glucose = plt.figure()
    ax3 = sns.scatterplot(x='Usia', y='TekananDarah', data=df, hue='Outcome', palette='magma')
    ax4 = sns.scatterplot(x=user_data['usia'], y=user_data['td'], s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 220, 10))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_td)

    # Usia vs Ketebalan Kulit
    st.header('Blood Pressure Value Graph (Others vs Yours)')
    fig_bp = plt.figure()
    ax5 = sns.scatterplot(x='Usia', y='KetebalanKulit', data=df, hue='Outcome', palette='Reds')
    ax6 = sns.scatterplot(x=user_data['usia'], y=user_data['kk'], s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 130, 10))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_kk)

    # Usia vs Insulin
    st.header('Skin Thickness Value Graph (Others vs Yours)')
    fig_st = plt.figure()
    ax7 = sns.scatterplot(x='Usia', y='Insulin', data=df, hue='Outcome', palette='Blues')
    ax8 = sns.scatterplot(x=user_data['usia'], y=user_data['insulin'], s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 110, 10))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_insulin)

    # Usia vs BMI
    st.header('Insulin Value Graph (Others vs Yours)')
    fig_i = plt.figure()
    ax9 = sns.scatterplot(x='Usia', y='BMI', data=df, hue='Outcome', palette='rocket')
    ax10 = sns.scatterplot(x=user_data['usia'], y=user_data['bmi'], s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 900, 50))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_bmi)

# OUTPUT
st.subheader('Your Report: ')
output = ''
if user_result[0] == 0:
    output = 'You are not Diabetic'
else:
    output = 'You are Diabetic'
st.title(output)
st.subheader('Accuracy: ')
st.write(str(accuracy_score(y_test, kmeans.predict(x_test)) * 100) + '%')
