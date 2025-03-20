import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import psycopg2




st.title("Welcome to WILL ETP")
st.write("Kindly upload your file")

# Function to get database connection dynamically
@st.cache_resource  # Ensures the connection is reused
def get_db_engine(db_name):
    try:
        if db_name == "ny_taxi":
            db_url = 'postgresql://root:root@pg-database:5432/ny_taxi'
        else:
            db_url = 'postgresql://root:root@pg-database2:5433/db2'
        return create_engine(db_url)
    except SQLAlchemyError as e:
        st.error(f"Database connection error: {e}")
        return None

# Database selection
option = st.selectbox("Select your database", ("ny_taxi", "db2"))
st.write(f"You selected: **{option}**")

# File upload
uploaded_file = st.file_uploader("Upload your spreadsheet", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        # Convert file to DataFrame
        file_extension = uploaded_file.name.split(".")[-1]

        if file_extension == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_extension == "xlsx":
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format!")
            st.stop()

        # Preview dataset
        st.write("Preview of uploaded file:")
        st.dataframe(df.head())

        # Get database connection
        engine = get_db_engine(option)

        if engine:
            # Show progress indicator
            with st.spinner(f"Uploading data to `{option}` database..."):
                df.to_sql(name='uploaded_data', con=engine, if_exists='append', index=False)
                st.success(f"Data uploaded successfully to `{option}` database!")
    except Exception as e:
        
