import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

sns.set_style('whitegrid')
plt.rcParams['font.size'] = 10

st.set_page_config(page_title="Sales Forecasting System", layout="wide")

@st.cache_data
def load_and_process_data():
    df = pd.read_csv('train.csv')
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], dayfirst=True)
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month
    df['Quarter'] = df['Order Date'].dt.quarter
    df['Shipping_Days'] = (df['Ship Date'] - df['Order Date']).dt.days
    return df

@st.cache_data
def get_monthly_data(df):
    monthly = df.resample('ME', on='Order Date')['Sales'].sum().reset_index()
    monthly.columns = ['Month_Start', 'Monthly_Sales']
    return monthly

@st.cache_data
def get_weekly_data(df):
    weekly = df.resample('W-Mon', on='Order Date')['Sales'].sum().reset_index()
    weekly.columns = ['Week_Start', 'Weekly_Sales']
    return weekly

@st.cache_data
def run_anomaly_detection(weekly_df):
    anom = weekly_df.copy()
    anom.set_index('Week_Start', inplace=True)
    iso = IsolationForest(contamination=0.05, random_state=42)
    anom['IF_Anomaly'] = iso.fit_predict(anom[['Weekly_Sales']])
    anom['Rolling_Mean'] = anom['Weekly_Sales'].rolling(8, center=True).mean()
    anom['Rolling_Std'] = anom['Weekly_Sales'].rolling(8, center=True).std()
    anom['Z_Score'] = (anom['Weekly_Sales'] - anom['Rolling_Mean']) / anom['Rolling_Std']
    anom['ZS_Anomaly'] = np.where(anom['Z_Score'].abs() > 2, -1, 1)
    return anom

@st.cache_data
def run_clustering(df):
    cluster_data = df.groupby('Sub-Category').agg({'Sales': ['sum', 'mean', 'std']}).round(2)
    cluster_data.columns = ['Total_Sales', 'Avg_Order_Value', 'Sales_Volatility']
    cluster_data = cluster_data.reset_index()
    subcat_yearly = df.groupby(['Sub-Category', 'Year'])['Sales'].sum().unstack()
    if len(subcat_yearly.columns) >= 2:
        fy, ly = subcat_yearly.columns[0], subcat_yearly.columns[-1]
        cluster_data['Growth_Rate'] = cluster_data['Sub-Category'].map(
            lambda sc: ((subcat_yearly.loc[sc, ly] - subcat_yearly.loc[sc, fy]) / subcat_yearly.loc[sc, fy]) * 100
            if sc in subcat_yearly.index and subcat_yearly.loc[sc, fy] > 0 else 0)
    else:
        cluster_data['Growth_Rate'] = 0
    feats = ['Total_Sales', 'Sales_Volatility', 'Avg_Order_Value', 'Growth_Rate']
    X = cluster_data[feats].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_data['Cluster'] = kmeans.fit_predict(X_scaled)
    labels = {0: 'High Volume, Stable Demand', 1: 'Growing Demand', 2: 'Low Volume, High Volatility', 3: 'Declining Demand'}
    cluster_data['Segment'] = cluster_data['Cluster'].map(labels)
    pca = PCA(n_components=2)
    pca_vals = pca.fit_transform(X_scaled)
    cluster_data['PCA1'] = pca_vals[:, 0]
    cluster_data['PCA2'] = pca_vals[:, 1]
    return cluster_data

@st.cache_data
def run_prophet_forecast(monthly_df, periods=3):
    pdf = monthly_df.rename(columns={'Month_Start': 'ds', 'Monthly_Sales': 'y'})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, seasonality_mode='multiplicative')
    model.fit(pdf)
    future = model.make_future_dataframe(periods=periods, freq='ME')
    forecast = model.predict(future)
    return forecast, model

df = load_and_process_data()
monthly = get_monthly_data(df)
weekly = get_weekly_data(df)
anom = run_anomaly_detection(weekly)
cluster_data = run_clustering(df)
forecast, prophet_model = run_prophet_forecast(monthly, 3)

st.sidebar.title("Sales Forecasting System")
page = st.sidebar.radio("Navigate", ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Segments"])

if page == "Sales Overview":
    st.title("Sales Overview Dashboard")

    col1, col2, col3 = st.columns(3)
    with col1:
        total_sales = df['Sales'].sum()
        st.metric("Total Sales", f"${total_sales:,.0f}")
    with col2:
        avg_order = df['Sales'].mean()
        st.metric("Avg Order Value", f"${avg_order:.2f}")
    with col3:
        total_orders = len(df)
        st.metric("Total Orders", f"{total_orders:,}")

    st.subheader("Monthly Sales Trend")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(monthly['Month_Start'], monthly['Monthly_Sales'], color='#2c3e50', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales ($)')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.subheader("Sales by Year")
    yearly = df.groupby('Year')['Sales'].sum()
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(yearly.index.astype(str), yearly.values, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'])
    for bar, val in zip(bars, yearly.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000, f'${val:,.0f}', ha='center', fontsize=10)
    ax.set_xlabel('Year')
    ax.set_ylabel('Total Sales ($)')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Sales by Region")
        region_data = df.groupby('Region')['Sales'].sum().sort_values()
        fig, ax = plt.subplots(figsize=(8, 4))
        colors_region = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
        bars = ax.barh(region_data.index, region_data.values, color=colors_region)
        for bar, val in zip(bars, region_data.values):
            ax.text(bar.get_width() + 1000, bar.get_y() + bar.get_height()/2, f'${val:,.0f}', va='center', fontsize=9)
        ax.set_xlabel('Sales ($)')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with col_b:
        st.subheader("Sales by Category")
        cat_data = df.groupby('Category')['Sales'].sum().sort_values()
        fig, ax = plt.subplots(figsize=(8, 4))
        colors_cat = ['#2ecc71', '#3498db', '#e74c3c']
        ax.pie(cat_data.values, labels=cat_data.index, autopct='%1.1f%%', colors=colors_cat, startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

    st.subheader("Interactive Filters")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        selected_region = st.selectbox("Select Region", ["All"] + sorted(df['Region'].unique()))
    with col_f2:
        selected_category = st.selectbox("Select Category", ["All"] + sorted(df['Category'].unique()))

    filtered = df.copy()
    if selected_region != "All":
        filtered = filtered[filtered['Region'] == selected_region]
    if selected_category != "All":
        filtered = filtered[filtered['Category'] == selected_category]

    st.write(f"Filtered Sales: ${filtered['Sales'].sum():,.0f} ({len(filtered)} orders)")
    fig, ax = plt.subplots(figsize=(12, 3))
    monthly_f = filtered.resample('ME', on='Order Date')['Sales'].sum()
    ax.plot(monthly_f.index, monthly_f.values, color='#e74c3c', linewidth=2)
    ax.set_title(f'Sales Trend: {selected_region} / {selected_category}')
    ax.set_ylabel('Sales ($)')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

elif page == "Forecast Explorer":
    st.title("Forecast Explorer")

    st.subheader("3-Month Sales Forecast (Prophet)")
    fig1 = prophet_model.plot(forecast)
    plt.title('Sales Forecast - Next 3 Months', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Sales ($)')
    st.pyplot(fig1)

    st.subheader("Forecast Components")
    fig2 = prophet_model.plot_components(forecast)
    st.pyplot(fig2)

    st.subheader("Forecast by Segment")
    segment_type = st.radio("Select Segment Type", ["Category", "Region"], horizontal=True)
    if segment_type == "Category":
        segments = ['Furniture', 'Technology', 'Office Supplies']
    else:
        segments = ['West', 'East', 'Central', 'South']

    horizon = st.slider("Forecast Horizon (months)", 1, 3, 3)

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    seg_data = {}
    for i, seg in enumerate(segments):
        if segment_type == "Category":
            sdf = df[df['Category'] == seg]
        else:
            sdf = df[df['Region'] == seg]
        monthly_s = sdf.resample('ME', on='Order Date')['Sales'].sum().reset_index()
        monthly_s.columns = ['ds', 'y']
        monthly_s = monthly_s[monthly_s['y'] > 0]
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, seasonality_mode='multiplicative')
        model.fit(monthly_s)
        future = model.make_future_dataframe(periods=horizon, freq='ME')
        fcst = model.predict(future)
        seg_data[seg] = fcst.tail(horizon)
        ax.plot(fcst.tail(horizon)['ds'], fcst.tail(horizon)['yhat'], marker='o', linewidth=2.5, label=seg, color=colors[i % len(colors)])
        ax.fill_between(fcst.tail(horizon)['ds'], fcst.tail(horizon)['yhat_lower'], fcst.tail(horizon)['yhat_upper'], color=colors[i % len(colors)], alpha=0.1)

    ax.set_title(f'{segment_type}-Level Forecast ({horizon} Months)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Forecasted Sales ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Forecast Values")
    for seg, fcst in seg_data.items():
        st.write(f"**{seg}**")
        for _, row in fcst.iterrows():
            st.write(f"  {row['ds'].strftime('%b %Y')}: ${row['yhat']:,.0f} (${row['yhat_lower']:,.0f} - ${row['yhat_upper']:,.0f})")

    train = monthly.rename(columns={'Month_Start': 'ds', 'Monthly_Sales': 'y'})
    train_set = train.iloc[:-3]
    test_set = train.iloc[-3:]
    eval_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, seasonality_mode='multiplicative')
    eval_model.fit(train_set)
    future_eval = eval_model.make_future_dataframe(periods=3, freq='ME')
    eval_fcst = eval_model.predict(future_eval)
    preds = eval_fcst.tail(3)['yhat'].values
    actuals = test_set['y'].values
    mae = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    st.subheader("Model Performance Metrics")
    col_m1, col_m2 = st.columns(2)
    col_m1.metric("MAE (Mean Absolute Error)", f"${mae:,.0f}")
    col_m2.metric("RMSE (Root Mean Squared Error)", f"${rmse:,.0f}")

elif page == "Anomaly Report":
    st.title("Anomaly Detection Report")

    st.subheader("Anomaly Detection: Isolation Forest vs Z-Score")
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    ax1 = axes[0]
    n = anom[anom['IF_Anomaly'] == 1]
    a = anom[anom['IF_Anomaly'] == -1]
    ax1.plot(n.index, n['Weekly_Sales'], 'o-', color='#3498db', markersize=3, linewidth=0.5, label='Normal', alpha=0.6)
    ax1.scatter(a.index, a['Weekly_Sales'], color='#e74c3c', s=100, marker='x', zorder=5, label='Anomaly')
    ax1.set_title('Isolation Forest', fontweight='bold')
    ax1.set_ylabel('Weekly Sales ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2 = axes[1]
    nz = anom[anom['ZS_Anomaly'] == 1]
    az = anom[anom['ZS_Anomaly'] == -1]
    ax2.plot(nz.index, nz['Weekly_Sales'], 'o-', color='#3498db', markersize=3, linewidth=0.5, label='Normal', alpha=0.6)
    ax2.scatter(az.index, az['Weekly_Sales'], color='#e74c3c', s=100, marker='x', zorder=5, label='Anomaly')
    ax2.set_title('Z-Score (|Z| > 2)', fontweight='bold')
    ax2.set_ylabel('Weekly Sales ($)')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Detected Anomaly Weeks")
    st.write("**Isolation Forest Anomalies:**")
    if_anom = anom[anom['IF_Anomaly'] == -1][['Weekly_Sales']].copy()
    if_anom.index = if_anom.index.strftime('%Y-%m-%d')
    if_anom.columns = ['Sales ($)']
    st.dataframe(if_anom.style.format({'Sales ($)': '${:,.2f}'}))

    st.write("**Z-Score Anomalies:**")
    zs_anom = anom[anom['ZS_Anomaly'] == -1][['Weekly_Sales', 'Z_Score']].copy()
    zs_anom.index = zs_anom.index.strftime('%Y-%m-%d')
    zs_anom.columns = ['Sales ($)', 'Z-Score']
    st.dataframe(zs_anom.style.format({'Sales ($)': '${:,.2f}', 'Z-Score': '{:.2f}'}))

    st.subheader("Anomaly Explanations")
    st.info("""
    - **November spikes**: Black Friday / Cyber Monday holiday sales
    - **December spikes**: Christmas / holiday season shopping
    - **September spikes**: Back-to-school / early holiday prep
    - **January drops**: Post-holiday demand slump
    - **February drops**: Post-holiday slowdown / inventory transitions
    """)

elif page == "Product Segments":
    st.title("Product Demand Segments")

    st.subheader("Cluster Visualization (PCA Reduced)")
    colors = {'High Volume, Stable Demand': '#2ecc71', 'Growing Demand': '#3498db', 'Low Volume, High Volatility': '#e74c3c', 'Declining Demand': '#f39c12'}
    fig, ax = plt.subplots(figsize=(12, 7))
    for seg in cluster_data['Segment'].unique():
        mask = cluster_data['Segment'] == seg
        ax.scatter(cluster_data.loc[mask, 'PCA1'], cluster_data.loc[mask, 'PCA2'],
                   c=colors.get(seg, '#333'), label=seg, s=200, alpha=0.7, edgecolors='black', linewidth=1)
        for _, row in cluster_data[mask].iterrows():
            ax.annotate(row['Sub-Category'], (row['PCA1'], row['PCA2']), fontsize=8, alpha=0.8, xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_title('Product Demand Segments', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Products by Segment")
    for seg in ['High Volume, Stable Demand', 'Growing Demand', 'Low Volume, High Volatility', 'Declining Demand']:
        items = cluster_data[cluster_data['Segment'] == seg]
        if not items.empty:
            st.write(f"**{seg}** ({len(items)} products)")
            st.dataframe(items[['Sub-Category', 'Total_Sales', 'Growth_Rate', 'Sales_Volatility']].style.format({
                'Total_Sales': '${:,.0f}', 'Growth_Rate': '{:+.1f}%', 'Sales_Volatility': '${:,.0f}'
            }))

    st.subheader("Recommended Stocking Strategies")
    st.success("""
    **High Volume, Stable Demand**: Maintain high safety stock, auto-reorder systems  
    **Growing Demand**: Increase order quantities, negotiate supplier contracts for volume  
    **Low Volume, High Volatility**: Just-in-time ordering, avoid bulk purchases  
    **Declining Demand**: Clear inventory, reduce reorder points, consider discontinuing
    """)

st.sidebar.markdown("---")
st.sidebar.info("Built with Streamlit | Sales Forecasting System")
