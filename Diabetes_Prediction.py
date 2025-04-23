import pickle
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# Title and Introduction with Emojis
st.title('🍩 Diabetes Prediction 🍩')

df = pd.read_csv(r"C:\Users\aresa\Downloads\diabetes.csv")


df_group = df.groupby('Outcome').mean()

# Load the model
model_path = r'C:\Users\aresa\OneDrive\Desktop\Machine Learning\diabetes_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Load the scaler
scaler_path = r'C:\Users\aresa\OneDrive\Desktop\Machine Learning\scaler.pkl'
with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

# Sidebar for navigation with color
select = st.sidebar.radio('🔎 Select an option', ['Predict', 'Explore'])

if select == 'Predict':
    # Input fields with labels
    st.write("Enter the required values to predict whether the person is diabetic or not. 🩺")
    pregnancies = st.selectbox("🤰 Number of Pregnancies",[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])  
    glucose = st.number_input("🩸 Glucose Level", min_value=0.0, step=0.001, format="%.0f")  
    blood_pressure = st.number_input("💓 Blood Pressure", min_value=0.0, step=0.001, format="%.0f")
    skin_thickness = st.number_input("🧑‍⚕️ Skin Thickness", min_value=0.0, step=0.001, format="%.0f")
    insulin = st.number_input("💉 Insulin Level", min_value=0.0, step=0.001, format="%.0f")
    bmi = st.number_input("🏋️ BMI", min_value=0.0, step=0.001, format="%.1f")
    dpf = st.number_input("🧬 Diabetes Pedigree Function", min_value=0.0, step=0.001, format="%.3f")
    age = st.number_input("👵 Age", min_value=0.0, step=1.0, format="%.0f")

    input_data = (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age)
    
    # Warning for unrealistic input values
    if all(val == 0 for val in input_data):
        st.warning("⚠️ Please make sure that all input values are realistic. Zero values are not valid for these features. ⚠️")
    else:
        if st.button('🔮 Predict'):
            # Convert input data to NumPy array and reshape
            input_data = np.asarray(input_data).reshape(1, -1)

            # Scale the input data
            transformed_data = scaler.transform(input_data)

            # Make a prediction
            prediction_model = model.predict(transformed_data)

            # Display the result with color
            if prediction_model[0] == 0:
                st.markdown('<p style="color:green; font-size:20px;">✅ The person is NOT diabetic. </p>', unsafe_allow_html=True)
            else:
                st.markdown('<p style="color:red; font-size:20px;">❌ The person is DIABETIC. </p>', unsafe_allow_html=True)

if select == 'Explore':
    st.header(":mag_right: Explore Data")

    # Add some text or further instructions
    st.write("🔍 Use the options below to explore and visualize the dataset.")

    # Tabular Data
    if st.checkbox('📋 Show Tabular Data'):
        st.table(df.head(10))

    # Statistical Summary
    st.markdown('## 📊 Statistical Summary of the DataFrame')
    if st.checkbox('📈 Show Statistics'):
        st.table(df.describe())

    # Interpretation of Outcome
    st.markdown('## 🤔 Interpretation of Outcome')
    if st.checkbox('🔍 Outcome Analysis'):
        st.table(df_group)
        st.write('''
        - 🟢 **Outcome = 0 (NON-Diabetic)**  
        - 🔴 **Outcome = 1 (Diabetic)**  

        ### Key Observations:
        1️⃣ The average glucose level for non-diabetic individuals is **around 110**, while for diabetic individuals, it's significantly higher at **141**.  
        2️⃣ Diabetic individuals generally have higher values in features like **BMI, blood pressure, and age**, indicating a strong correlation between these factors and diabetes.  
        3️⃣ Non-diabetic individuals, on the other hand, have values closer to the normal range, which suggests they are at a lower risk.
        ''')

    # Correlation Graph
    st.markdown('## 🔗 Correlation Heatmap')
    fig, ax = plt.subplots(figsize=(5, 2.5))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    st.pyplot(fig)

# Graph Options
 
    st.title('📈 Graphs')
    graph = st.selectbox('📊 Select Graph', ['Bar Graph', 'Scatter Plot', 'Histogram'])

# Bar Graph
    if graph == 'Bar Graph':
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.barplot(data=df, ax=ax, x='Outcome', y='Insulin')  # Explicitly set x and y axes
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)  # Rotate x-axis labels
        ax.set_title("📊 Bar Graph: Outcome vs Insulin", fontsize=14)  # Add a title
        ax.set_xlabel("Outcome", fontsize=12)  # Customize x-axis label
        ax.set_ylabel("Average Insulin", fontsize=12)  # Customize y-axis label
        st.pyplot(fig)

# Scatter Plot
    if graph == 'Scatter Plot':
        value = st.slider('🎚️ Filter Using Insulin', 0, 200, step=1)  # Add a slider to filter insulin values
        dff = df.loc[df['Insulin'] < value]  # Filter rows where 'Insulin' is less than the slider value

        fig, ax = plt.subplots(figsize=(5, 2))  # Set the plot size
        sns.scatterplot(data=dff, x='Outcome', y='Insulin', ax=ax)  # Use the filtered DataFrame (dff)
        ax.set_title(f"📍 Scatter Plot (Insulin < {value})", fontsize=14)  # Add a title to indicate filtering
        ax.set_xlabel("Outcome", fontsize=12)  # Customize x-axis label
        ax.set_ylabel("Insulin", fontsize=12)  # Customize y-axis label
        st.pyplot(fig)  # Render the plot in Streamlit

# Histogram
    if graph == 'Histogram':
        fig, ax = plt.subplots(figsize=(5, 2))  # Create a figure and axis
        sns.histplot(data=df, x='Glucose', kde=True, ax=ax)  # Use sns.histplot, add KDE
        ax.set_title("📊 Histogram of Glucose Levels", fontsize=14)  # Add a title
        ax.set_xlabel("Glucose", fontsize=12)  # Label for the x-axis
        ax.set_ylabel("Frequency", fontsize=12)  # Label for the y-axis
        st.pyplot(fig)
