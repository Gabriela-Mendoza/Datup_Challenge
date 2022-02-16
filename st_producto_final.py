import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from pmdarima.arima import auto_arima

# Título
st.sidebar.markdown("### Predicción de ventas al próximo mes")

# Lectura de Dataframe tratado
base = pd.read_csv("ventas_nuevas_tratada.csv", parse_dates = ["Mes"])

# Toma códigos únicos de producto
productos = np.unique(base["Código Material"])

# Selección de código de producto
producto = st.sidebar.selectbox("Código de producto", tuple(productos))

# Filtro de DataFrame
df = base[base["Código Material"] == producto]
df = df.set_index("Mes")
df = pd.DataFrame(df["Venta Neta Kilos"])

# Título de la página
st.markdown("### Predicción para el producto de código " + str(producto))

model = auto_arima(df.values)

# Mejor modelo
st.markdown("#### Mejor modelo obtenido")
st.write("ARIMA" + str(model.get_params()["order"]) + str(model.get_params()["seasonal_order"]))


# Plot de predicción 

st.markdown("#### Plot de Predicción")
# Forecast
n_periods = 4
fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = np.arange(len(df.values), len(df.values)+n_periods)

# make series for plotting purpose
fc_series = pd.Series(fc, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# Plot
fig = plt.figure(figsize=(10.7,7.5))
plt.plot(df.values)
plt.plot(fc_series, color='tab:orange')
plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)

plt.title("Final Forecast of WWW Usage", loc = "left")
st.pyplot(fig)

# Plot de predicción 

st.markdown("#### Predicción para el próximo mes")
st.write(np.round(fc[0],2))

# Plots de diagnóstico
st.markdown("#### Plots de Diagnóstico")
fig = model.plot_diagnostics(figsize=(10.7,7.5))
st.pyplot(fig)

# Sumario de modelos
st.markdown("#### Sumário de Modelo")
st.write(model.summary())

# st.write() - Print
# st.markdown - Markdown
# st.dataframe y st.table - DataFrames
# st.plotly_chart (Plots de Plotly); st.pyplot(Matplotlib, seaborn) - Tiene que emplear sobre una figura (fig)