
import streamlit as st
import pandas as pd
from neuralprophet import NeuralProphet
import matplotlib.pyplot as plt

st.set_page_config(page_title="Deep AI Forecast - Sales & Customers", layout="wide")

st.title("🧠 Deep Learning AI Forecasting")
st.markdown("Upload your historical data to forecast **Sales** and **Customers** using NeuralProphet (AI model).")

uploaded_file = st.file_uploader("📤 Upload your Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("📊 Raw Data Preview")
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

    st.subheader("📥 Download Forecasts")
    export_df = pd.DataFrame({
        'Date': forecast_sales['ds'],
        'Forecasted Sales': forecast_sales['yhat'],
        'Forecasted Customers': forecast_customers['yhat']
    })

    st.dataframe(export_df.tail(14))

    csv = export_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download CSV", csv, "forecast_results.csv", "text/csv")
else:
    st.info("Please upload your dataset.")
