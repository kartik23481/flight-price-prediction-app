# .\new_flight_venv_latest\Scripts\Activate.ps1


# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)

# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# from datetime import datetime

# # ✅ Import your custom transformers and functions
# from feature_utils import (
#     is_north, find_part_of_month, part_of_day,
#     make_month_object, remove_duration, have_info,
#     duration_category
# )
# from rbf import RBFPercentileSimilarity

# # ✅ Load models only once
# @st.cache_resource
# def load_models():
#     column_transformer = joblib.load("column_transformer.joblib")
#     final_model = joblib.load("xgb_flight_price_model.joblib")
#     return column_transformer, final_model

# column_transformer, final_model = load_models()

# # 🎨 Streamlit app UI
# st.title("✈️ Flight Price Prediction App")

# # ✅ Collect user inputs
# airline = st.text_input("Enter airline")
# source = st.text_input("Enter source")
# destination = st.text_input("Enter destination")
# duration = st.number_input("Enter duration in minutes (e.g. 120)", min_value=0)
# total_stops = st.number_input("Enter total stops (e.g. 0,1,2..)", min_value=0)
# additional_info = st.text_input("Enter additional info")
# dtoj_day = st.number_input("Enter day of journey (1-31)", min_value=1, max_value=31)
# dtoj_month = st.number_input("Enter month of journey (1-12)", min_value=1, max_value=12)
# dept_time_hour = st.number_input("Enter departure hour (0-23)", min_value=0, max_value=23)

# # ✅ Prediction button
# if st.button("Predict Flight Price"):
#     # 🔗 Create input dataframe
#     dtoj_year = 2019  # for consistency

#     input_df = pd.DataFrame({
#         'airline': [airline],
#         'source': [source],
#         'destination': [destination],
#         'duration': [duration],
#         'total_stops': [total_stops],
#         'additional_info': [additional_info],
#         'dep_time_hour': [dept_time_hour],
#         'dtoj_day': [dtoj_day],
#         'dtoj_month': [dtoj_month],
#         'dtoj_year': [dtoj_year],
#     })

#     # ✅ Create date column and is_weekend
#     input_df = input_df.assign(
#         date = pd.to_datetime(
#             input_df.rename(columns={'dtoj_year': 'year', 
#                                      'dtoj_month': 'month', 
#                                      'dtoj_day': 'day'})[['year', 'month', 'day']]
#         )
#     )
#     input_df = input_df.assign(is_weekend = input_df['date'].dt.weekday >= 5)

#     # ✅ Drop dtoj_year
#     input_df.drop(columns=['dtoj_year'], inplace=True)

#     # 📝 Show input dataframe
#     st.write("Input DataFrame:", input_df)

#     # 🔮 Transform and predict
#     input_df_transformed = column_transformer.transform(input_df)
#     prediction = final_model.predict(input_df_transformed)

#     st.success(f"💰 Predicted Flight Price: ₹{prediction[0]:,.2f}")

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
import sys

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

# ✅ Import your custom transformers and functions
from utils.feature_utils import (
    is_north, find_part_of_month, part_of_day,
    make_month_object, remove_duration, have_info,
    duration_category
)
from utils.rbf import RBFPercentileSimilarity

# 🎨 Page config
st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="centered"
)

# ✅ Load models only once
@st.cache_resource
def load_models():
    column_transformer_path = os.path.join("artifacts", "column_transformer.joblib")
    models_path = os.path.join( "artifacts", "models")

    xgboostmodel_path = os.path.join(models_path,"xgb_flight_price_model.joblib")
    # column_transformer = joblib.load("column_transformer.joblib")
    column_transformer = joblib.load(column_transformer_path)
    xgb_model = joblib.load(xgboostmodel_path)
    return column_transformer, xgb_model

column_transformer, final_model = load_models()

# ✨ App title and description
st.markdown(
    "<h1 style='text-align: center; color: #4A90E2;'>✈️ Flight Price Prediction</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Predict your flight ticket prices instantly and plan smarter.</p>",
    unsafe_allow_html=True
)

st.divider()

# ✅ Input form for better UX
with st.form("flight_input_form"):
    st.markdown("### 📝 Enter Flight Details")

    # Use columns for clean layout
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### ✈️ Airline & Source")
        airline = st.text_input("Airline", placeholder="e.g. Indigo")
        source = st.text_input("Source", placeholder="e.g. Delhi")
        duration = st.number_input("Duration (minutes)", min_value=0, placeholder="e.g. 120")

    with col2:
        st.markdown("#### 🛬 Destination & Stops")
        destination = st.text_input("Destination", placeholder="e.g. Mumbai")
        total_stops = st.number_input("Total Stops", min_value=0, placeholder="e.g. 1")
        additional_info = st.text_input("Additional Info", placeholder="e.g. No info")

    st.markdown("#### 🗓️ Date & Time")
    col3, col4 = st.columns(2)
    with col3:
        date_input = st.date_input("Date of Journey", value=datetime(2019,6,1))
    with col4:
        dept_time_hour = st.number_input("Departure Hour (0-23)", min_value=0, max_value=23)

    submitted = st.form_submit_button("🔮 Predict Flight Price")

    if submitted:
        # ✅ Extract day and month
        dtoj_day = date_input.day
        dtoj_month = date_input.month
        dtoj_year = 2019  # for consistency

        # 🔗 Create input dataframe
        input_df = pd.DataFrame({
            'airline': [airline],
            'source': [source],
            'destination': [destination],
            'duration': [duration],
            'total_stops': [total_stops],
            'additional_info': [additional_info],
            'dep_time_hour': [dept_time_hour],
            'dtoj_day': [dtoj_day],
            'dtoj_month': [dtoj_month],
            'dtoj_year': [dtoj_year],
        })

        # ✅ Create date column and is_weekend
        input_df = input_df.assign(
            date = pd.to_datetime(
                input_df.rename(columns={'dtoj_year': 'year', 
                                         'dtoj_month': 'month', 
                                         'dtoj_day': 'day'})[['year', 'month', 'day']]
            )
        )
        input_df = input_df.assign(is_weekend = input_df['date'].dt.weekday >= 5)

        # ✅ Drop dtoj_year
        input_df.drop(columns=['dtoj_year'], inplace=True)

        # 🔮 Transform and predict
        input_df_transformed = column_transformer.transform(input_df)
        prediction = final_model.predict(input_df_transformed)

        st.divider()

        # 💰 Display result beautifully
        st.markdown(f"""
        <div style="background-color: #DFF0D8; padding: 20px; border-radius: 10px;">
            <h3 style="color: #3C763D;">💰 Predicted Flight Price: ₹{prediction[0]:,.2f}</h3>
        </div>
        """, unsafe_allow_html=True)

        # 📝 Show input dataframe neatly
        with st.expander("🔍 View Processed Input Data"):
            st.write(input_df)

        # 🎯 Placeholder for future insights and graphs
        st.markdown("### 📊 Insights & Recommendations (Coming Soon)")
        st.info("Graphs and recommendations based on historical data will appear here in future versions.")

        # 💡 Tip
        st.info("📈 *Tip: Booking early or on weekdays often leads to cheaper prices.*")

st.divider()

# ✅ About section with minimal branding
st.markdown("""
<div style="text-align: center;">
    <p style="font-size: 14px;">Developed by <b>Kartik Srivastava</b> | <a href="https://github.com" target="_blank">GitHub</a> | <a href="https://linkedin.com" target="_blank">LinkedIn</a></p>
</div>
""", unsafe_allow_html=True)
