
import streamlit as st
import pandas as pd
from neuralprophet import NeuralProphet
import matplotlib.pyplot as plt

# Sample credentials
USER_CREDENTIALS = {"admin": "mcdosan2025"}

# Session state for login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login_screen():
    st.image("deep_sales_forecast/assets/mcdonalds_logo.png", width=150)
    st.title("🔐 McDonald's San Carlos Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Welcome, {username}!")
        else:
            st.error("Invalid credentials")

def main_app():
    st.title("🍟 McDonald's San Carlos AI Sales Forecast")
    st.markdown("Welcome to the deep learning forecaster. Upload your data below.")
    uploaded_file = st.file_uploader("📤 Upload your Excel file", type=["xlsx"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.subheader("📊 Data Preview")
        st.dataframe(df.head())

        df['Date'] = pd.to_datetime(df['Date'])
        df['unique_id'] = 'store_1'

        df_model_sales = df[['Date', 'Sales']].rename(columns={'Date': 'ds', 'Sales': 'y'})
        df_model_sales['unique_id'] = 'sales'

        df_model_customers = df[['Date', 'Customers']].rename(columns={'Date': 'ds', 'Customers': 'y'})
        df_model_customers['unique_id'] = 'customers'

        model_sales = NeuralProphet()
        model_sales.fit(df_model_sales, freq='D')
        future_sales = model_sales.make_future_dataframe(df_model_sales, periods=14)
        forecast_sales = model_sales.predict(future_sales)

        model_customers = NeuralProphet()
        model_customers.fit(df_model_customers, freq='D')
        future_customers = model_customers.make_future_dataframe(df_model_customers, periods=14)
        forecast_customers = model_customers.predict(future_customers)

        st.subheader("📈 Sales Forecast")
        fig_sales = model_sales.plot(forecast_sales)
        st.pyplot(fig_sales)

        st.subheader("👥 Customer Forecast")
        fig_customers = model_customers.plot(forecast_customers)
        st.pyplot(fig_customers)

        st.subheader("📥 Download Forecast")
        export_df = pd.DataFrame({
            "Date": forecast_sales['ds'],
            "Forecasted Sales": forecast_sales['yhat'],
            "Forecasted Customers": forecast_customers['yhat']
        })
        st.dataframe(export_df.tail(14))
        csv = export_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download CSV", csv, "forecast_results.csv", "text/csv")

    st.sidebar.write(f"👤 Logged in as: {st.session_state.get('username', '')}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.experimental_rerun()

# App flow
if st.session_state.logged_in:
    main_app()
else:
    login_screen()
