import streamlit as st
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(page_title="Token Analytics Dashboard", layout="wide")
st.title("📊 Token Usage Analytics Dashboard")

@st.cache_data
def read_token_logs(file_paths):
    """
    Read multiple token log files and create a combined dataframe.
    Supports both old format and new format with agent_work metadata.
    """
    data = []
    
    for file_path in file_paths:
        if not Path(file_path).exists():
            continue
            
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Parse JSON log entry
                    log_entry = json.loads(line)
                    
                    # Extract timestamp (first key)
                    timestamp_str = list(log_entry.keys())[0]
                    timestamp = pd.to_datetime(timestamp_str)
                    
                    # Get the model data
                    model_data = log_entry.get(timestamp_str, {})
                    if isinstance(model_data, dict):
                        # Get first model's data
                        model_key = list(model_data.keys())[0]
                        tokens = model_data[model_key]
                        
                        # Extract agent_work if it exists (new format)
                        agent_work = log_entry.get('agent_work', 'Unknown')
                        processing_time = log_entry.get('processing_time', log_entry.get('processing__time', 0))
                        
                        # Extract platform from log entry
                        platform = log_entry.get('Platform', 'Unknown')
                        
                        row = {
                            'timestamp': timestamp,
                            'date': timestamp.date(),
                            'time': timestamp.time(),
                            'model': model_key,
                            'input_tokens': tokens.get('input_tokens', 0),
                            'output_tokens': tokens.get('output_tokens', 0),
                            'total_tokens': tokens.get('total_tokens', 0),
                            'process': platform,
                            'agent_work': agent_work,
                            'processing_time': processing_time,
                            'source_file': Path(file_path).name
                        }
                        data.append(row)
                except (json.JSONDecodeError, KeyError, IndexError, ValueError):
                    continue
    
    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values('timestamp')
    return df

# Load data
log_files = [
    'Token_usage_log.txt'
]

df = read_token_logs(log_files)

if df.empty:
    st.error("❌ No data found. Please check the log files.")
    st.stop()

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Date range filter
date_filter_type = st.sidebar.radio(
    "Select Date Range:",
    ["All Dates", "Specific Date", "Date Range"]
)

if date_filter_type == "All Dates":
    filtered_df = df.copy()
    date_range_text = f"All ({df['date'].min()} to {df['date'].max()})"
elif date_filter_type == "Specific Date":
    selected_date = st.sidebar.date_input(
        "Select Date:",
        value=pd.to_datetime(df['date'].max()),
        min_value=pd.to_datetime(df['date'].min()),
        max_value=pd.to_datetime(df['date'].max())
    )
    filtered_df = df[df['date'] == selected_date]
    date_range_text = f"{selected_date}"
else:  # Date Range
    min_date = pd.to_datetime(df['date'].min())
    max_date = pd.to_datetime(df['date'].max())
    date_range = st.sidebar.date_input(
        "Select Date Range:",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        date_range_text = f"{start_date} to {end_date}"
    else:
        filtered_df = df.copy()
        date_range_text = "Select both dates"

# Process/Work Type filter
available_processes = sorted(filtered_df['process'].unique())
selected_processes = st.sidebar.multiselect(
    "Select Work Types:",
    available_processes,
    default=available_processes
)

# If no processes selected, select all
if not selected_processes:
    selected_processes = available_processes

filtered_df = filtered_df[filtered_df['process'].isin(selected_processes)]

# Agent Work filter
available_agent_works = sorted(filtered_df['agent_work'].unique())
selected_agent_works = st.sidebar.multiselect(
    "Select Agent Work:",
    available_agent_works,
    default=available_agent_works
)

# If no agent works selected, select all
if not selected_agent_works:
    selected_agent_works = available_agent_works

filtered_df = filtered_df[filtered_df['agent_work'].isin(selected_agent_works)]

# Time filter (optional)
use_time_filter = st.sidebar.checkbox("Filter by Time?")
if use_time_filter:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_time = st.time_input("Start Time:", value=datetime.min.time())
    with col2:
        end_time = st.time_input("End Time:", value=datetime.max.time())
    
    filtered_df = filtered_df[
        (filtered_df['time'] >= start_time) & 
        (filtered_df['time'] <= end_time)
    ]

# Display summary statistics
st.sidebar.markdown("---")
st.sidebar.header("📈 Summary Stats")
if not filtered_df.empty:
    st.sidebar.metric("Total Entries", len(filtered_df))
    st.sidebar.metric("Total Input Tokens", f"{filtered_df['input_tokens'].sum():,}")
    st.sidebar.metric("Total Output Tokens", f"{filtered_df['output_tokens'].sum():,}")
    st.sidebar.metric("Total Tokens", f"{filtered_df['total_tokens'].sum():,}")

# Main content
st.header(f"Data for: {date_range_text}")

if filtered_df.empty:
    st.warning("⚠️ No data available for the selected filters.")
    st.stop()

# Tabs for different views
tab1, tab2, tab3 = st.tabs(["📊 Token Summary", "📈 Time Series", "📋 Table View"])

with tab1:
    st.subheader("Token Type Distribution")
    
    # Summary by work type
    work_type_summary = filtered_df.groupby('process').agg({
        'input_tokens': 'sum',
        'output_tokens': 'sum',
        'total_tokens': 'sum'
    }).reset_index()
    
    # Create grouped bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=work_type_summary['process'],
        y=work_type_summary['input_tokens'],
        name='Input Tokens',
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        x=work_type_summary['process'],
        y=work_type_summary['output_tokens'],
        name='Output Tokens',
        marker_color='lightcoral'
    ))
    
    fig.add_trace(go.Bar(
        x=work_type_summary['process'],
        y=work_type_summary['total_tokens'],
        name='Total Tokens',
        marker_color='lightgreen'
    ))
    
    fig.update_layout(
        barmode='group',
        title="Token Usage by Work Type",
        xaxis_title="Work Type",
        yaxis_title="Token Count",
        hovermode='x unified',
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Horizontal bar chart for each token type separately
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig_input = px.bar(
            work_type_summary,
            y='process',
            x='input_tokens',
            orientation='h',
            title='Input Tokens',
            color='input_tokens',
            color_continuous_scale='Blues'
        )
        fig_input.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_input, use_container_width=True)
    
    with col2:
        fig_output = px.bar(
            work_type_summary,
            y='process',
            x='output_tokens',
            orientation='h',
            title='Output Tokens',
            color='output_tokens',
            color_continuous_scale='Reds'
        )
        fig_output.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_output, use_container_width=True)
    
    with col3:
        fig_total = px.bar(
            work_type_summary,
            y='process',
            x='total_tokens',
            orientation='h',
            title='Total Tokens',
            color='total_tokens',
            color_continuous_scale='Greens'
        )
        fig_total.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_total, use_container_width=True)

with tab2:
    st.subheader("Token Usage Over Time")
    
    # Prepare data for time series
    time_series = filtered_df.groupby(['timestamp', 'process']).agg({
        'input_tokens': 'sum',
        'output_tokens': 'sum',
        'total_tokens': 'sum'
    }).reset_index()
    
    # Select which token type to display
    token_type = st.radio(
        "Select Token Type:",
        ["Total Tokens", "Input Tokens", "Output Tokens"],
        horizontal=True
    )
    
    token_column = {
        "Total Tokens": "total_tokens",
        "Input Tokens": "input_tokens",
        "Output Tokens": "output_tokens"
    }[token_type]
    
    fig_ts = px.line(
        time_series,
        x='timestamp',
        y=token_column,
        color='process',
        title=f"{token_type} Over Time",
        markers=True,
        height=500
    )
    
    fig_ts.update_layout(
        xaxis_title="Timestamp",
        yaxis_title="Token Count",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_ts, use_container_width=True)

with tab3:
    st.subheader("Raw Data Table")
    
    display_df = filtered_df[[
        'timestamp', 'date', 'time', 'process', 'model', 'agent_work',
        'input_tokens', 'output_tokens', 'total_tokens'
    ]].copy()
    
    display_df = display_df.sort_values('timestamp', ascending=False)
    
    st.dataframe(display_df, use_container_width=True)
    
    # Download button
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name=f"token_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
