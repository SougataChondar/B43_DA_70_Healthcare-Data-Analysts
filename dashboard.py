import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("C:/Users/ASUS/Downloads/New folder (5)/lung cancer data.csv")

# Encode categorical values
df['GENDER'] = df['GENDER'].map({'M': 1, 'F': 0})
df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

# Split features and target
X = df.drop(columns=['LUNG_CANCER'])
y = df['LUNG_CANCER']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump((model, scaler), "lung_cancer_model.pkl")

# Streamlit UI
st.set_page_config(page_title = "Lung Cancer Perdiction Dashboard", page_icon=":hospital:", layout= "wide")
st.title(":hospital: Lung Cancer Prediction Dashboard")

st.markdown("---")
st.write("Enter patient details to predict lung cancer risk.")

# Data Visualization
st.subheader("Data Insights")



# User inputs
st.sidebar.image("logo.jpg", width=280)
st.sidebar.title("🔍 Filtes")

# Interactive filters
selected_gender = st.sidebar.selectbox("Filter by Gender", ["All", "Male", "Female"])
selected_age = st.sidebar.slider("Filter by Age", int(df['AGE'].min()), int(df['AGE'].max()), (int(df['AGE'].min()), int(df['AGE'].max())))

# Apply filters
df_filtered = df.copy()
if selected_gender != "All":
    df_filtered = df_filtered[df_filtered['GENDER'] == (1 if selected_gender == "Male" else 0)]
df_filtered = df_filtered[(df_filtered['AGE'] >= selected_age[0]) & (df_filtered['AGE'] <= selected_age[1])]

# Gender distribution
fig2 = px.bar(df_filtered, x=df_filtered['GENDER'].map({1: 'Male', 0: 'Female'}), color='LUNG_CANCER', 
               title="Lung Cancer Distribution by Gender", barmode='group', color_continuous_scale='blues', template="plotly_dark")
st.plotly_chart(fig2)


# Age distribution
fig1 = px.histogram(df_filtered, x='AGE', nbins=20, title="Age Distribution of Patients", color_discrete_sequence=['blue'], template="plotly_dark")
st.plotly_chart(fig1)


# Smoking vs Lung Cancer
fig3 = px.bar(df_filtered, x='SMOKING', color='LUNG_CANCER', 
               title="Effect of Smoking on Lung Cancer", barmode='group', color_continuous_scale='blues', template="plotly_dark")
st.plotly_chart(fig3)


st.markdown("---")
age = st.number_input("Age", min_value=20, max_value=100, value=50)
gender = st.selectbox("Gender", ["Male", "Female"])
smoking = st.slider(" Smoking History", 0, 2, 1)

yellow_fingers = st.slider("Yellow Fingers", 0, 2, 1)
anxiety = st.slider("Anxiety", 0, 2, 1)
peer_pressure = st.slider("Peer Pressure", 0, 2, 1)
chronic_disease = st.slider("Chronic Disease", 0, 2, 1)
fatigue = st.slider("Fatigue", 0, 2, 1)
allergy = st.slider("Allergy", 0, 2, 1)
wheezing = st.slider("Wheezing", 0, 2, 1)
alcohol_consuming = st.slider("Alcohol Consuming", 0, 2, 1)
coughing = st.slider("Coughing", 0, 2, 1)
shortness_of_breath = st.slider("Shortness of Breath", 0, 2, 1)
swallowing_difficulty = st.slider("Swallowing Difficulty", 0, 2, 1)
chest_pain = st.slider("Chest Pain", 0, 2, 1)

# Convert gender to numeric
gender_num = 1 if gender == "Male" else 0

# Prepare input data
input_data = np.array([[
    gender_num, age, smoking, yellow_fingers, anxiety, peer_pressure,
    chronic_disease, fatigue, allergy, wheezing, alcohol_consuming,
    coughing, shortness_of_breath, swallowing_difficulty, chest_pain
]])

# Load model and make prediction
model, scaler = joblib.load("lung_cancer_model.pkl")
input_data_scaled = scaler.transform(input_data)
prediction = model.predict(input_data_scaled)

# Display result
if st.button("Predict"):
    st.write("Prediction: ", "Lung Cancer Detected :heavy_exclamation_mark: " if prediction[0] == 1 else "No Lung Cancer :white_check_mark: ")


st.write("---")
# Footer Section
st.markdown(
    """
    <footer style='text-align: center'>
    Project Code: [B43_DA_70_Healthcare Data Analysts] | Data Source: [kaggle](https://www.kaggle.com/datasets/iamtanmayshukla/lung-cancer-data?resource=download)
    
    Credits:
    This project was collaboratively developed and executed by :

    -> Sougata Chondar - fd39_255

    </footer>
    """,
    unsafe_allow_html=True
)
