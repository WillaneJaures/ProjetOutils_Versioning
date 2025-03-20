import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC

# Title
st.title("Upload and Train a Machine Learning Model")
st.write("Please upload a CSV or Excel file.")

# File uploader
upload_file = st.file_uploader("Upload your file", type=['csv', 'xlsx'])

if upload_file is not None:
    try:
        # Load the data
        file_extension = upload_file.name.split(".")[-1]
        if file_extension == 'csv':
            df = pd.read_csv(upload_file)
        elif file_extension == 'xlsx':
            df = pd.read_excel(upload_file)
        else:
            st.error("Unsupported file format")
            st.stop()

        # Show preview
        st.write("Preview of the file:")
        st.dataframe(df.head())

        # Select target variable
        target = st.selectbox("Select the target column:", df.columns)

        # Identify if it's a regression or classification problem
        is_classification = df[target].nunique() <= 2  # Binary classification if 2 unique values

        # Select features (numerical columns only)
        features = st.multiselect("Select features:", df.select_dtypes(include=['int64', 'float64']).columns)
        
        if not features:
            st.error("Please select at least one feature.")
            st.stop()

        # Handle missing values
        df = df.dropna()

        # Split data
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Choose model
        model_options = ["Linear Regression", "Random Forest", "Logistic Regression", "SVM (Classification)"] if is_classification else ["Linear Regression", "Random Forest"]
        model_choice = st.selectbox("Select your Machine Learning model:", model_options)

        # Train the selected model
        if model_choice == "Linear Regression":
            model = LinearRegression()
        elif model_choice == "Random Forest":
            model = RandomForestRegressor() if not is_classification else RandomForestClassifier()
        elif model_choice == "Logistic Regression":
            model = LogisticRegression()
        elif model_choice == "SVM (Classification)":
            model = SVC()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Show results based on model type
        if is_classification:
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"**Accuracy:** {accuracy:.4f}")
        else:
            mse = mean_squared_error(y_test, y_pred)
            st.write(f"**Mean Squared Error:** {mse:.4f}")

        # Plot results
        if not is_classification:  # Regression Plot
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title(f"Actual vs. Predicted Values ({model_choice})")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")
