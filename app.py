# app.py
# Restyled Streamlit dashboard (keeps original logic & calculations)
# Purpose: Visual refresh and reorganized UI while preserving data logic and calculations.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import datetime

st.set_page_config(page_title="Uber Ride Analytics", layout="wide")

# ---------- Helper / Data Loading (unchanged logic, lightly reformatted) ----------
@st.cache_data(show_spinner=True)
def load_main(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    # datetime parsing fallback logic preserved
    if 'datetime' in df.columns:
        try:
            df['datetime'] = pd.to_datetime(df['datetime'])
        except:
            if 'Date' in df.columns and 'Time' in df.columns:
                df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='coerce')
            else:
                df['datetime'] = pd.to_datetime(df.get('Date', None), errors='coerce')
    else:
        if 'Date' in df.columns and 'Time' in df.columns:
            df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='coerce')
        else:
            df['datetime'] = pd.to_datetime(df.get('Date', None), errors='coerce')

    for col in ['Booking Value', 'Ride Distance', 'Driver Ratings', 'Customer Rating']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['month_name'] = df['datetime'].dt.strftime('%b')
    df['weekday'] = df['datetime'].dt.day_name()
    df['week_of_year'] = df['datetime'].dt.isocalendar().week
    df['is_weekend'] = df['weekday'].isin(['Saturday', 'Sunday'])
    if 'Booking Status' in df.columns:
        df['Booking Status'] = df['Booking Status'].astype(str).str.strip().str.title()
    return df

@st.cache_data(show_spinner=True)
def load_coords(path, pickup=True):
    coords = pd.read_csv(path)
    coords.columns = [c.strip() for c in coords.columns]
    if 'Pickup Location' in coords.columns:
        coords = coords.rename(columns={'Pickup Location': 'Location'})
    elif 'Drop Location' in coords.columns:
        coords = coords.rename(columns={'Drop Location': 'Location'})
    elif 'Location' not in coords.columns:
        coords = coords.rename(columns={coords.columns[0]: 'Location'})

    lat_cols = [c for c in coords.columns if 'lat' in c.lower()]
    lon_cols = [c for c in coords.columns if 'lon' in c.lower()]
    if lat_cols:
        coords = coords.rename(columns={lat_cols[0]: 'Latitude'})
    if lon_cols:
        coords = coords.rename(columns={lon_cols[0]: 'Longitude'})
    coords['Location'] = coords['Location'].astype(str).str.strip()
    return coords[['Location', 'Latitude', 'Longitude']]

# ------------------- User paths (update these if needed) -------------------
MAIN_PATH = r"C:\Users\Saiteja\Desktop\vs projects\dashboard\group3_dataset.csv"
PICKUP_COORDS_PATH = r"C:\Users\Saiteja\Desktop\vs projects\dashboard\pickup_location_coords_delhi.csv"
DROP_COORDS_PATH = r"C:\Users\Saiteja\Desktop\vs projects\dashboard\drop_location_coords_delhi.csv"

# Load data
with st.spinner("Loading dataset..."):
    df = load_main(MAIN_PATH)
    pickup_coords = load_coords(PICKUP_COORDS_PATH, pickup=True)
    drop_coords = load_coords(DROP_COORDS_PATH, pickup=False)

# Merge coords
if 'Pickup Location' in df.columns:
    df['Pickup Location'] = df['Pickup Location'].astype(str).str.strip()
if 'Drop Location' in df.columns:
    df['Drop Location'] = df['Drop Location'].astype(str).str.strip()

df = df.merge(
    pickup_coords.rename(columns={'Location': 'Pickup Location', 'Latitude':'Pickup_Latitude', 'Longitude':'Pickup_Longitude'}),
    on='Pickup Location', how='left'
)

df = df.merge(
    drop_coords.rename(columns={'Location': 'Drop Location', 'Latitude':'Drop_Latitude', 'Longitude':'Drop_Longitude'}),
    on='Drop Location', how='left'
)

# ------------------- Restyled Sidebar (compact & grouped) -------------------
st.sidebar.title("🔎 Filters & Settings")

with st.sidebar.expander("Date & Sampling", expanded=True):
    min_date = df['date'].min()
    max_date = df['date'].max()
    date_range = st.date_input("Date range", [min_date, max_date], min_value=min_date, max_value=max_date)
    sample_option = st.selectbox("Sampling for maps/charts (keeps UI snappy)", options=['No sampling','Sample 2k','Sample 5k'], index=1)

with st.sidebar.expander("Ride & Vehicle Filters", expanded=True):
    vehicles = sorted(df['Vehicle Type'].dropna().unique().tolist()) if 'Vehicle Type' in df.columns else []
    selected_vehicles = st.multiselect("Vehicle Type (multi-select)", options=vehicles, default=vehicles)

    statuses = sorted(df['Booking Status'].dropna().unique().tolist()) if 'Booking Status' in df.columns else []
    selected_statuses = st.multiselect("Booking Status (multi)", options=statuses, default=statuses)

with st.sidebar.expander("Location Filters", expanded=False):
    pickups = sorted(df['Pickup Location'].dropna().unique().tolist()) if 'Pickup Location' in df.columns else []
    drops = sorted(df['Drop Location'].dropna().unique().tolist()) if 'Drop Location' in df.columns else []
    selected_pickup_location = st.selectbox("Pickup Location", options=['All']+pickups, index=0)
    selected_drop_location = st.selectbox("Drop Location", options=['All']+drops, index=0)

with st.sidebar.expander("Value & Metrics", expanded=False):
    if 'Booking Value' in df.columns:
        min_bv = float(np.nan_to_num(df['Booking Value'].min()))
        max_bv = float(np.nan_to_num(df['Booking Value'].max()))
    else:
        min_bv, max_bv = 0.0, 1000.0
    bv_range = st.slider("Booking Value range", min_value=min_bv, max_value=max_bv, value=(min_bv, max_bv))

st.sidebar.caption("UI tip: use the filters to narrow results before heavy visualizations.")

# ------------------- Apply filters (keeps your original mask logic but adapted for multi-selects) -------------------
mask = pd.Series(True, index=df.index)

# date
mask_date_range = (df['date'] >= date_range[0]) & (df['date'] <= date_range[1])
if date_range[0] == df['date'].min() and date_range[1] == df['date'].max():
    mask = mask & (mask_date_range | df['date'].isna())
else:
    mask = mask & mask_date_range

# vehicles (multi)
if 'Vehicle Type' in df.columns and selected_vehicles:
    mask = mask & df['Vehicle Type'].isin(selected_vehicles)

# statuses (multi)
if 'Booking Status' in df.columns and selected_statuses:
    mask = mask & df['Booking Status'].isin(selected_statuses)

# pickup/drop
if selected_pickup_location and selected_pickup_location != 'All':
    mask = mask & df['Pickup Location'].isin([selected_pickup_location])
if selected_drop_location and selected_drop_location != 'All':
    mask = mask & df['Drop Location'].isin([selected_drop_location])

# booking value
if 'Booking Value' in df.columns:
    mask_bv = df['Booking Value'].between(bv_range[0], bv_range[1], inclusive='both')
    if bv_range[0] == float(np.nan_to_num(df['Booking Value'].min())) and bv_range[1] == float(np.nan_to_num(df['Booking Value'].max())):
        mask = mask & (mask_bv | df['Booking Value'].isna())
    else:
        mask = mask & mask_bv

# filtered df
df_f = df[mask].copy()

# ------------------- Top-level header + KPIs in a card-like row -------------------
st.title("🚗 Uber Ride Analytics — 2024 (Restyled)")
st.markdown("A refreshed UI while keeping your analysis unchanged.")

k1, k2, k3, k4 = st.columns([1.2,1,1,1])

total_bookings = len(df_f)
completed_count = len(df_f[df_f['Booking Status'].str.contains('Completed', na=False)]) if 'Booking Status' in df_f.columns else np.nan
cancelled_count = len(df_f[df_f['Booking Status'].str.contains('Cancel', na=False)]) if 'Booking Status' in df_f.columns else np.nan
avg_booking_value = df_f['Booking Value'].mean() if 'Booking Value' in df_f.columns else np.nan

k1.metric(label="Total bookings", value=f"{total_bookings:,}")
k2.metric(label="Completed", value=f"{int(completed_count):,}" if not np.isnan(completed_count) else "N/A",
          delta=f"{(completed_count/total_bookings*100):.1f}%" if (not np.isnan(completed_count) and total_bookings>0) else None)
k3.metric(label="Cancelled", value=f"{int(cancelled_count):,}" if not np.isnan(cancelled_count) else "N/A",
          delta=f"{(cancelled_count/total_bookings*100):.1f}%" if (not np.isnan(cancelled_count) and total_bookings>0) else None)
k4.metric(label="Avg Booking Value", value=(f"₹{avg_booking_value:,.2f}" if not np.isnan(avg_booking_value) else "N/A"))

st.divider()

# ------------------- Main content organized into tabs for clearer navigation -------------------
tabs = st.tabs(["Overview", "Maps & Hotspots", "Behavior & Ratings", "Data Export"])

# ---------- Overview Tab ----------
with tabs[0]:
    st.header("Overview — Trends & Patterns")
    c1, c2 = st.columns([2,1])
    with c1:
        # Time of day demand
        df_f['time_of_day'] = pd.cut(
            df_f['hour'].fillna(0),
            bins=[-0.1, 5, 11, 16, 20, 23],
            labels=['Late Night', 'Morning', 'Afternoon', 'Evening', 'Night'],
            ordered=True
        )
        # observed=False to keep current behavior and silence FutureWarning
        tod_counts = df_f.groupby('time_of_day', observed=False).size().reset_index(name='count')
        fig_tod = px.bar(tod_counts, x='time_of_day', y='count', title="Demand by Time of Day (binned)", text='count')
        st.plotly_chart(fig_tod, use_container_width=True)

        # Weekday bookings
        weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        weekday_counts = (
            df_f.groupby('weekday', observed=False).size()
            .reindex(weekday_order)
            .reset_index(name='count')
        )
        fig_week = px.bar(weekday_counts, x='weekday', y='count', title="Bookings by Weekday", text='count')
        st.plotly_chart(fig_week, use_container_width=True)

    with c2:
        # Monthly trend
        monthly = df_f.groupby(['month','month_name'], observed=False).size().reset_index(name='count').sort_values('month')
        fig_month = px.line(monthly, x='month_name', y='count', markers=True, title="Monthly Booking Trend")
        st.plotly_chart(fig_month, use_container_width=True)

    st.markdown("**Top booking days (spikes)** — click to expand")
    with st.expander("Top 10 days by bookings", expanded=False):
        top_days = df_f.groupby('date').size().reset_index(name='bookings').sort_values('bookings', ascending=False).head(10)
        st.dataframe(top_days)

# ---------- Maps & Hotspots Tab ----------
with tabs[1]:
    st.header("Maps & Hotspots — Delhi NCR")
    map_col1, map_col2 = st.columns(2)
    with map_col1:
        st.subheader("Pickup Hotspots")
        pickup_map_df = df_f[['Pickup_Latitude','Pickup_Longitude']].dropna()
        if not pickup_map_df.empty:
            st.map(pickup_map_df.rename(columns={'Pickup_Latitude':'lat','Pickup_Longitude':'lon'}), zoom=10)
            top_pickups = df_f['Pickup Location'].value_counts().reset_index(name='count').rename(columns={'index':'Pickup Location'})
            st.dataframe(top_pickups.head(20))
        else:
            st.info("No pickup coordinates available after merge.")

    with map_col2:
        st.subheader("Dropoff Hotspots")
        drop_map_df = df_f[['Drop_Latitude','Drop_Longitude']].dropna()
        if not drop_map_df.empty:
            st.map(drop_map_df.rename(columns={'Drop_Latitude':'lat','Drop_Longitude':'lon'}), zoom=10)
            top_drops = df_f['Drop Location'].value_counts().reset_index(name='count').rename(columns={'index':'Drop Location'})
            st.dataframe(top_drops.head(20))
        else:
            st.info("No dropoff coordinates available after merge.")

    st.markdown("---")
    st.subheader("Sampled Scatter Geo (quick preview)")
    sample_n = 2000 if sample_option == 'Sample 2k' else 5000 if sample_option == 'Sample 5k' else len(df_f)
    sample_n = min(sample_n, len(df_f))
    sample = df_f.sample(sample_n, random_state=42) if sample_n < len(df_f) else df_f

    if not sample[['Pickup_Latitude','Pickup_Longitude']].dropna().empty:
        fig = px.scatter_geo(sample.dropna(subset=['Pickup_Latitude','Pickup_Longitude']),
                             lat='Pickup_Latitude', lon='Pickup_Longitude', scope='asia',
                             hover_name='Pickup Location', title='Sampled Pickup locations')
        st.plotly_chart(fig, use_container_width=True)

    if not sample[['Drop_Latitude','Drop_Longitude']].dropna().empty:
        fig2 = px.scatter_geo(sample.dropna(subset=['Drop_Latitude','Drop_Longitude']),
                              lat='Drop_Latitude', lon='Drop_Longitude', scope='asia',
                              hover_name='Drop Location', title='Sampled Dropoff locations')
        st.plotly_chart(fig2, use_container_width=True)

# ---------- Behavior & Ratings Tab ----------
with tabs[2]:
    st.header("Customer & Driver Behaviour")
    if 'Customer ID' in df_f.columns:
        cust_counts = df_f.groupby('Customer ID').size().reset_index(name='bookings')
        one_time = (cust_counts['bookings'] == 1).sum()
        repeat = (cust_counts['bookings'] > 1).sum()
        cust_df = pd.DataFrame({'segment': ['One-time','Repeat'], 'count':[one_time, repeat]})
        fig_custseg = px.pie(cust_df, names='segment', values='count', title='Customer Segments: One-time vs Repeat', hole=0.35)
        st.plotly_chart(fig_custseg, use_container_width=True)

        cust_merged = df_f.merge(cust_counts, on='Customer ID', how='left')
        cust_merged['segment'] = np.where(cust_merged['bookings']>1, 'Repeat', 'One-time')
        if 'Booking Status' in cust_merged.columns:
            seg_success = cust_merged.groupby('segment')['Booking Status'].apply(lambda s: s.str.contains('Completed', na=False).mean()).reset_index(name='completion_rate')
            st.write("Completion rate by customer segment:")
            st.dataframe(seg_success)

    else:
        st.info("Customer-level analysis requires a 'Customer ID' column.")

    st.markdown("---")
    st.subheader("Satisfaction & Ratings")
    if 'Customer Rating' in df_f.columns:
        fig_rating = px.histogram(df_f, x='Customer Rating', nbins=20, title='Customer Rating Distribution')
        st.plotly_chart(fig_rating, use_container_width=True)

    if 'Booking Status' in df_f.columns and 'Customer Rating' in df_f.columns:
        df_r = df_f.copy()
        df_r['is_cancelled'] = df_r['Booking Status'].str.contains('Cancel', na=False)
        rating_cancel = df_r.groupby('is_cancelled')['Customer Rating'].median().reset_index()
        st.write("Median customer rating for cancelled vs non-cancelled bookings:")
        st.dataframe(rating_cancel)

# ---------- Data Export Tab ----------
with tabs[3]:
    st.header("Filtered Data & Export")
    st.write("Filtered rows:", len(df_f))
    st.dataframe(df_f.head(200))

    @st.cache_data
    def convert_df_to_csv(df_local):
        return df_local.to_csv(index=False).encode('utf-8')

    csv = convert_df_to_csv(df_f)
    st.download_button(label="Download filtered data as CSV", data=csv, file_name='uber_filtered.csv', mime='text/csv')

    st.markdown("---")
    st.caption("Dashboard generated from: group3_dataset.csv, pickup_location_coords_delhi.csv, drop_location_coords_delhi.csv")

# ---------- Notes (kept minimal) ----------
with st.expander("Notes & Assumptions (click to open)"):
    st.markdown("""
    - UI reorganized into tabs: Overview, Maps & Hotspots, Behavior & Ratings, Data Export.
    - Filters moved into grouped expanders in the sidebar and allow multi-select for Vehicle and Booking Status.
    - Core data parsing & calculations are preserved; visual arrangement changed only.
    - For heavy datasets consider sampling or narrowing date range for interactive performance.
    """)

# End
