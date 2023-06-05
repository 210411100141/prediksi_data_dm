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

df = pd.read_csv("dm.csv", delimiter=";")

# HEADINGS
st.title('Prediksi Diabetes')
st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')
st.write(df.describe())

tab1, tab2 = st.columns([2, 1])

with tab1:
    st.image("ar.png")

with tab2:
    # X AND Y DATA
    x = df.drop(['Outcome'], axis=1)
    y = df['Outcome']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # FUNCTION
    def user_report():
        usia = st.sidebar.slider('Usia', 21, 88, 33)
        glukosa = st.sidebar.slider('Glukosa', 0, 200, 120)
        tekanan_darah = st.sidebar.slider('Tekanan Darah', 0, 122, 70)
        ketebalan_kulit = st.sidebar.slider('Ketebalan Kulit', 0, 100, 20)
        insulin = st.sidebar.slider('Insulin', 0, 846, 79)
        bmi = st.sidebar.slider('BMI', 0, 67, 20)
        dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)

        user_report_data = {
            'Usia': usia,
            'Glukosa': glukosa,
            'Tekanan Darah': tekanan_darah,
            'Ketebalan Kulit': ketebalan_kulit,
            'Insulin': insulin,
            'BMI': bmi,
            'Diabetes Pedigree Function': dpf
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

    # Age vs Glucose
    st.header('Glucose Value Graph (Others vs Yours)')
    fig_glucose = plt.figure()
    ax1 = sns.scatterplot(x='Usia', y='Glukosa', data=df, hue='Outcome', palette='magma')
    ax2 = sns.scatterplot(x=user_data['Usia'], y=user_data['Glukosa'], s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 220, 10))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_glucose)

    # Age vs Blood Pressure
    st.header('Blood Pressure Value Graph (Others vs Yours)')
    fig_bp = plt.figure()
    ax3 = sns.scatterplot(x='Usia', y='Tekanan Darah', data=df, hue='Outcome', palette='Reds')
    ax4 = sns.scatterplot(x=user_data['Usia'], y=user_data['Tekanan Darah'], s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 130, 10))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_bp)

    # Age vs Skin Thickness
    st.header('Skin Thickness Value Graph (Others vs Yours)')
    fig_st = plt.figure()
    ax5 = sns.scatterplot(x='Usia', y='Ketebalan Kulit', data=df, hue='Outcome', palette='Blues')
    ax6 = sns.scatterplot(x=user_data['Usia'], y=user_data['Ketebalan Kulit'], s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 110, 10))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_st)

    # Age vs Insulin
    st.header('Insulin Value Graph (Others vs Yours)')
    fig_i = plt.figure()
    ax7 = sns.scatterplot(x='Usia', y='Insulin', data=df, hue='Outcome', palette='rocket')
    ax8 = sns.scatterplot(x=user_data['Usia'], y=user_data['Insulin'], s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 900, 50))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_i)

    # Age vs BMI
    st.header('BMI Value Graph (Others vs Yours)')
    fig_bmi = plt.figure()
    ax9 = sns.scatterplot(x='Usia', y='BMI', data=df, hue='Outcome', palette='rainbow')
    ax10 = sns.scatterplot(x=user_data['Usia'], y=user_data['BMI'], s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 70, 5))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_bmi)

    # Age vs DPF
    st.header('DPF Value Graph (Others vs Yours)')
    fig_dpf = plt.figure()
    ax11 = sns.scatterplot(x='Usia', y='Diabetes Pedigree Function', data=df, hue='Outcome', palette='YlOrBr')
    ax12 = sns.scatterplot(x=user_data['Usia'], y=user_data['Diabetes Pedigree Function'], s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 3, 0.2))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_dpf)

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

