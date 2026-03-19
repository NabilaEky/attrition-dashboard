import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="Attrition Dashboard", layout="wide")

st.title("📊 Employee Attrition Dashboard")
st.write("Dashboard untuk menganalisis faktor-faktor yang mempengaruhi attrition karyawan")

# Load Data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/main/employee/employee_data.csv"
    return pd.read_csv(url)

df = load_data()


# Prepocessing
if df['Attrition'].dtype == 'object':
    df['Attrition'] = df['Attrition'].map({
        'No': 0,
        'Yes': 1
    })

# drop Attrition 
df = df.dropna(subset=['Attrition'])

# tipe data
df['Attrition'] = df['Attrition'].astype(int)

# dashboard
df_model = df.copy()

# label 
df_model['Attrition'] = df_model['Attrition'].map({
    0: 'Tidak Keluar',
    1: 'Keluar'
})

# Ambil min max 
age_min = int(np.nanmin(df_model['Age']))
age_max = int(np.nanmax(df_model['Age']))

salary_min = int(np.nanmin(df_model['MonthlyIncome']))
salary_max = int(np.nanmax(df_model['MonthlyIncome']))

# Load model
try:
    st.write("Isi root:", os.listdir())

    if os.path.exists("model"):
        st.write("Isi folder model:", os.listdir("model"))
    else:
        st.write("Folder model tidak ditemukan")

    model = pickle.load(open('model/model.pkl', 'rb'))
    features = pickle.load(open('model/features.pkl', 'rb'))

    st.success("Model berhasil di-load")
    model_loaded = True

except Exception as e:
    st.error(f"Gagal load model: {e}")
    model_loaded = False


# KPI
st.markdown("## 📊 Ringkasan")

col1, col2 = st.columns(2)
col1.metric("Total Karyawan", len(df_model))
col2.metric("Attrition Rate", f"{(df_model['Attrition'] == 'Keluar').mean()*100:.2f}%")

# Distribusi Attrition
col5, col6 = st.columns(2)

with col5:
    st.subheader("📌 Distribusi Attrition")
    st.bar_chart(df_model['Attrition'].value_counts())

with col6:
    st.subheader("📌 Pengaruh Job Satisfaction terhadap Attrition")
    st.bar_chart(pd.crosstab(df_model['JobSatisfaction'], df_model['Attrition']))

# Pengaruh Usia terhadap Attrition
col7, col8 = st.columns(2)

with col7:
    st.subheader("📌 Pengaruh Usia terhadap Attrition")
    st.bar_chart(pd.crosstab(df_model['Age'], df_model['Attrition']))

# Pengaruh Work-Life terhadap Balance
with col8:
    st.subheader("📌 Pengaruh Work-Life terhadap Balance")
    st.bar_chart(pd.crosstab(df_model['WorkLifeBalance'], df_model['Attrition']))

# Pengaruh Gaji terhadap Attrition
df_model['SalaryGroup'] = pd.qcut(
    df_model['MonthlyIncome'],
    4,
    labels=["Rendah", "Menengah", "Tinggi", "Sangat Tinggi"]
)

col9, col10 = st.columns(2)

with col9:
    st.subheader("📌 Pengaruh Gaji terhadap Attrition")
    st.bar_chart(pd.crosstab(df_model['SalaryGroup'], df_model['Attrition']))

# Pengaruh Masa Kerja terhadap Attrition
with col10:
    st.subheader("📌 Pengaruh Masa Kerja terhadap Attrition")
    st.bar_chart(pd.crosstab(df_model['YearsAtCompany'], df_model['Attrition']))

# Prediksi
st.subheader("🔮 Prediksi Attrition")

if model_loaded:

    col11, col12 = st.columns(2)

    with col11:
        job_satisfaction = st.selectbox("Job Satisfaction", [1,2,3,4])
        work_life = st.selectbox("Work Life Balance", [1,2,3,4])

    with col12:
        income = st.number_input("Monthly Income", 1000, 20000, 5000)
        years = st.number_input("Years at Company", 0, 40, 3)

    if st.button("Prediksi"):

        input_data = pd.DataFrame([{
            'JobSatisfaction': job_satisfaction,
            'WorkLifeBalance': work_life,
            'MonthlyIncome': income,
            'YearsAtCompany': years
        }])

        input_data = pd.get_dummies(input_data)
        input_data = input_data.reindex(columns=features, fill_value=0)

        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠️ Karyawan Berpotensi Keluar")
        else:
            st.success("✅ Karyawan Berpotensi Bertahan")

else:
    st.info("Model tidak tersedia, hanya dashboard yang ditampilkan")
