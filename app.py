# # .\new_flight_venv_latest\Scripts\Activate.ps1


# # import warnings
# # warnings.filterwarnings("ignore", category=UserWarning)

# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # import joblib
# # from datetime import datetime

# # # ✅ Import your custom transformers and functions
# # from feature_utils import (
# #     is_north, find_part_of_month, part_of_day,
# #     make_month_object, remove_duration, have_info,
# #     duration_category
# # )
# # from rbf import RBFPercentileSimilarity

# # # ✅ Load models only once
# # @st.cache_resource
# # def load_models():
# #     column_transformer = joblib.load("column_transformer.joblib")
# #     final_model = joblib.load("xgb_flight_price_model.joblib")
# #     return column_transformer, final_model

# # column_transformer, final_model = load_models()

# # # 🎨 Streamlit app UI
# # st.title("✈️ Flight Price Prediction App")

# # # ✅ Collect user inputs
# # airline = st.text_input("Enter airline")
# # source = st.text_input("Enter source")
# # destination = st.text_input("Enter destination")
# # duration = st.number_input("Enter duration in minutes (e.g. 120)", min_value=0)
# # total_stops = st.number_input("Enter total stops (e.g. 0,1,2..)", min_value=0)
# # additional_info = st.text_input("Enter additional info")
# # dtoj_day = st.number_input("Enter day of journey (1-31)", min_value=1, max_value=31)
# # dtoj_month = st.number_input("Enter month of journey (1-12)", min_value=1, max_value=12)
# # dept_time_hour = st.number_input("Enter departure hour (0-23)", min_value=0, max_value=23)

# # # ✅ Prediction button
# # if st.button("Predict Flight Price"):
# #     # 🔗 Create input dataframe
# #     dtoj_year = 2019  # for consistency

# #     input_df = pd.DataFrame({
# #         'airline': [airline],
# #         'source': [source],
# #         'destination': [destination],
# #         'duration': [duration],
# #         'total_stops': [total_stops],
# #         'additional_info': [additional_info],
# #         'dep_time_hour': [dept_time_hour],
# #         'dtoj_day': [dtoj_day],
# #         'dtoj_month': [dtoj_month],
# #         'dtoj_year': [dtoj_year],
# #     })

# #     # ✅ Create date column and is_weekend
# #     input_df = input_df.assign(
# #         date = pd.to_datetime(
# #             input_df.rename(columns={'dtoj_year': 'year', 
# #                                      'dtoj_month': 'month', 
# #                                      'dtoj_day': 'day'})[['year', 'month', 'day']]
# #         )
# #     )
# #     input_df = input_df.assign(is_weekend = input_df['date'].dt.weekday >= 5)

# #     # ✅ Drop dtoj_year
# #     input_df.drop(columns=['dtoj_year'], inplace=True)

# #     # 📝 Show input dataframe
# #     st.write("Input DataFrame:", input_df)

# #     # 🔮 Transform and predict
# #     input_df_transformed = column_transformer.transform(input_df)
# #     prediction = final_model.predict(input_df_transformed)

# #     st.success(f"💰 Predicted Flight Price: ₹{prediction[0]:,.2f}")

# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)
# import os
# import sys

# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# from datetime import datetime

# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

# # ✅ Import your custom transformers and functions
# from utils.feature_utils import (
#     is_north, find_part_of_month, part_of_day,
#     make_month_object, remove_duration, have_info,
#     duration_category
# )
# from utils.rbf import RBFPercentileSimilarity

# # 🎨 Page config
# st.set_page_config(
#     page_title="Flight Price Predictor",
#     page_icon="✈️",
#     layout="centered"
# )

# # ✅ Load models only once
# @st.cache_resource
# def load_models():
#     column_transformer_path = os.path.join("artifacts", "column_transformer.joblib")
#     models_path = os.path.join( "artifacts", "models")

#     xgboostmodel_path = os.path.join(models_path,"xgb_flight_price_model.joblib")
#     # column_transformer = joblib.load("column_transformer.joblib")
#     column_transformer = joblib.load(column_transformer_path)
#     xgb_model = joblib.load(xgboostmodel_path)
#     return column_transformer, xgb_model

# column_transformer, final_model = load_models()

# # ✨ App title and description
# st.markdown(
#     "<h1 style='text-align: center; color: #4A90E2;'>✈️ Flight Price Prediction</h1>",
#     unsafe_allow_html=True
# )
# st.markdown(
#     "<p style='text-align: center;'>Predict your flight ticket prices instantly and plan smarter.</p>",
#     unsafe_allow_html=True
# )

# st.divider()

# # ✅ Input form for better UX
# with st.form("flight_input_form"):
#     st.markdown("### 📝 Enter Flight Details")

#     # Use columns for clean layout
#     col1, col2 = st.columns(2)
#     with col1:
#         st.markdown("#### ✈️ Airline & Source")
#         airline = st.text_input("Airline", placeholder="e.g. Indigo")
#         source = st.text_input("Source", placeholder="e.g. Delhi")
#         duration = st.number_input("Duration (minutes)", min_value=0, placeholder="e.g. 120")

#     with col2:
#         st.markdown("#### 🛬 Destination & Stops")
#         destination = st.text_input("Destination", placeholder="e.g. Mumbai")
#         total_stops = st.number_input("Total Stops", min_value=0, placeholder="e.g. 1")
#         additional_info = st.text_input("Additional Info", placeholder="e.g. No info")

#     st.markdown("#### 🗓️ Date & Time")
#     col3, col4 = st.columns(2)
#     with col3:
#         date_input = st.date_input("Date of Journey", value=datetime(2019,6,1))
#     with col4:
#         dept_time_hour = st.number_input("Departure Hour (0-23)", min_value=0, max_value=23)

#     submitted = st.form_submit_button("🔮 Predict Flight Price")

#     if submitted:
#         # ✅ Extract day and month
#         dtoj_day = date_input.day
#         dtoj_month = date_input.month
#         dtoj_year = 2019  # for consistency

#         # 🔗 Create input dataframe
#         input_df = pd.DataFrame({
#             'airline': [airline],
#             'source': [source],
#             'destination': [destination],
#             'duration': [duration],
#             'total_stops': [total_stops],
#             'additional_info': [additional_info],
#             'dep_time_hour': [dept_time_hour],
#             'dtoj_day': [dtoj_day],
#             'dtoj_month': [dtoj_month],
#             'dtoj_year': [dtoj_year],
#         })

#         # ✅ Create date column and is_weekend
#         input_df = input_df.assign(
#             date = pd.to_datetime(
#                 input_df.rename(columns={'dtoj_year': 'year', 
#                                          'dtoj_month': 'month', 
#                                          'dtoj_day': 'day'})[['year', 'month', 'day']]
#             )
#         )
#         input_df = input_df.assign(is_weekend = input_df['date'].dt.weekday >= 5)

#         # ✅ Drop dtoj_year
#         input_df.drop(columns=['dtoj_year'], inplace=True)

#         # 🔮 Transform and predict
#         input_df_transformed = column_transformer.transform(input_df)
#         prediction = final_model.predict(input_df_transformed)

#         st.divider()

#         # 💰 Display result beautifully
#         st.markdown(f"""
#         <div style="background-color: #DFF0D8; padding: 20px; border-radius: 10px;">
#             <h3 style="color: #3C763D;">💰 Predicted Flight Price: ₹{prediction[0]:,.2f}</h3>
#         </div>
#         """, unsafe_allow_html=True)

#         # 📝 Show input dataframe neatly
#         with st.expander("🔍 View Processed Input Data"):
#             st.write(input_df)

#         # 🎯 Placeholder for future insights and graphs
#         st.markdown("### 📊 Insights & Recommendations (Coming Soon)")
#         st.info("Graphs and recommendations based on historical data will appear here in future versions.")

#         # 💡 Tip
#         st.info("📈 *Tip: Booking early or on weekdays often leads to cheaper prices.*")

# st.divider()

# # ✅ About section with minimal branding
# st.markdown("""
# <div style="text-align: center;">
#     <p style="font-size: 14px;">Developed by <b>Kartik Srivastava</b> | <a href="https://github.com" target="_blank">GitHub</a> | <a href="https://linkedin.com" target="_blank">LinkedIn</a></p>
# </div>
# """, unsafe_allow_html=True)




# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)
# import os
# import sys
# import streamlit as st
# import pandas as pd
# import joblib
# from datetime import datetime
# import time

# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

# from utils.feature_utils import (
#     is_north, find_part_of_month, part_of_day,
#     make_month_object, remove_duration, have_info,
#     duration_category
# )
# from utils.rbf import RBFPercentileSimilarity

# # 🎨 Page Config
# st.set_page_config(
#     page_title="Flight Price Predictor ✈️",
#     page_icon="✈️",
#     layout="wide"
# )

# # 📦 Load Models
# @st.cache_resource
# def load_models():
#     column_transformer_path = os.path.join("artifacts", "column_transformer.joblib")
#     models_path = os.path.join("artifacts", "models")
#     xgboostmodel_path = os.path.join(models_path, "xgb_flight_price_model.joblib")

#     column_transformer = joblib.load(column_transformer_path)
#     xgb_model = joblib.load(xgboostmodel_path)
#     return column_transformer, xgb_model

# column_transformer, final_model = load_models()

# # 🎨 Next.js-inspired CSS
# st.markdown("""
# <style>
# /* Full-screen gradient background */
# .stApp {
#     background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
#     color: white;
#     font-family: 'Segoe UI', sans-serif;
# }

# /* Glass effect cards */
# .glass-card {
#     background: rgba(255, 255, 255, 0.08);
#     border-radius: 16px;
#     padding: 30px;
#     backdrop-filter: blur(10px);
#     -webkit-backdrop-filter: blur(10px);
#     box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
# }

# /* Inputs */
# .stTextInput>div>div>input,
# .stNumberInput>div>input {
#     background: rgba(255,255,255,0.15);
#     border: none;
#     border-radius: 8px;
#     color: white;
# }
# .stTextInput>div>div>input::placeholder,
# .stNumberInput>div>input::placeholder {
#     color: #ccc;
# }

# /* Button */
# .stButton>button {
#     background: linear-gradient(90deg, #ff7e5f, #feb47b);
#     color: white;
#     font-weight: bold;
#     border-radius: 8px;
#     padding: 12px 20px;
#     border: none;
#     font-size: 16px;
# }
# .stButton>button:hover {
#     transform: scale(1.03);
#     background: linear-gradient(90deg, #feb47b, #ff7e5f);
#     transition: 0.3s;
# }

# /* Result text */
# .result-text {
#     font-size: 28px;
#     font-weight: bold;
#     color: #00e676;
#     text-shadow: 0px 0px 8px rgba(0, 230, 118, 0.8);
# }

# /* Center alignment */
# .center {
#     text-align: center;
# }
# </style>
# """, unsafe_allow_html=True)

# # 🏷️ Title
# st.markdown("<h1 class='center'>✈️ Flight Price Predictor</h1>", unsafe_allow_html=True)
# st.markdown("<p class='center'>A modern AI tool to predict your flight ticket prices instantly</p>", unsafe_allow_html=True)
# st.markdown("<br>", unsafe_allow_html=True)

# # 📋 Form in Glass Card
# with st.form("flight_input_form"):
#     st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
#     st.markdown("### 📝 Enter Flight Details")

#     col1, col2 = st.columns(2)
#     with col1:
#         airline = st.text_input("Airline", placeholder="e.g. Indigo")
#         source = st.text_input("Source", placeholder="e.g. Delhi")
#         duration = st.number_input("Duration (minutes)", min_value=0, placeholder="e.g. 120")

#     with col2:
#         destination = st.text_input("Destination", placeholder="e.g. Mumbai")
#         total_stops = st.number_input("Total Stops", min_value=0, placeholder="e.g. 1")
#         additional_info = st.text_input("Additional Info", placeholder="e.g. No info")

#     col3, col4 = st.columns(2)
#     with col3:
#         date_input = st.date_input("Date of Journey", value=datetime(2019, 6, 1))
#     with col4:
#         dept_time_hour = st.number_input("Departure Hour (0-23)", min_value=0, max_value=23)

#     submitted = st.form_submit_button("🔮 Predict Price")
#     st.markdown("</div>", unsafe_allow_html=True)

# # 🚀 On Submit
# if submitted:
#     dtoj_day = date_input.day
#     dtoj_month = date_input.month
#     dtoj_year = 2019

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

#     input_df = input_df.assign(
#         date=pd.to_datetime(
#             input_df.rename(columns={'dtoj_year': 'year', 
#                                      'dtoj_month': 'month', 
#                                      'dtoj_day': 'day'})[['year', 'month', 'day']]
#         )
#     )
#     input_df = input_df.assign(is_weekend=input_df['date'].dt.weekday >= 5)
#     input_df.drop(columns=['dtoj_year'], inplace=True)

#     input_df_transformed = column_transformer.transform(input_df)
#     prediction = final_model.predict(input_df_transformed)

#     # 🎯 Animated Result
#     st.markdown("<div class='glass-card center'>", unsafe_allow_html=True)
#     st.markdown("## 💰 Predicted Price")
#     placeholder = st.empty()
#     for i in range(0, int(prediction[0]), int(prediction[0]) // 50 or 1):
#         placeholder.markdown(f"<p class='result-text'>₹{i:,.0f}</p>", unsafe_allow_html=True)
#         time.sleep(0.02)
#     placeholder.markdown(f"<p class='result-text'>₹{prediction[0]:,.2f}</p>", unsafe_allow_html=True)
#     st.markdown("</div>", unsafe_allow_html=True)

#     # 📊 Tips
#     st.markdown("<br><div class='glass-card'>📈 Tip: Booking early, avoiding weekends, and picking early morning flights can save you money.</div>", unsafe_allow_html=True)



# # Working code

# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)
# import os
# import sys
# import streamlit as st
# import pandas as pd
# import joblib
# import time
# from datetime import datetime
# import plotly.express as px
# import plotly.graph_objects as go


# # Add utils path
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

# from utils.feature_utils import (
#     is_north, find_part_of_month, part_of_day,
#     make_month_object, remove_duration, have_info,
#     duration_category
# )
# from utils.rbf import RBFPercentileSimilarity

# # --------------------
# # PAGE CONFIG
# # --------------------
# st.set_page_config(
#     page_title="Flight Price Predictor",
#     page_icon="✈️",
#     layout="wide"
# )


# # --------------------
# # CUSTOM CSS
# # --------------------
# st.markdown("""
#     <style>
#         /* Black background */
#         .stApp {
#             background-color: black;
#         }
#         /* Input fields */
#         .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select, .stDateInput>div>div>input {
#             background: rgba(20,20,20,0.9);
#             color: #00ffcc;
#             border: 1px solid #00ffcc;
#             border-radius: 8px;
#         }
#         /* Labels */
#         label {
#             color: #00ffcc !important;
#             font-weight: bold;
#         }
#         /* Buttons */
#         .stButton>button {
#             background: linear-gradient(45deg, #00ffcc, #0066ff);
#             color: white;
#             font-weight: bold;
#             border-radius: 10px;
#             padding: 10px 20px;
#             box-shadow: 0 0 15px #00ffcc;
#             transition: transform 0.2s ease-in-out;
#             border: none;
#         }
#         .stButton>button:hover {
#             transform: scale(1.05);
#             box-shadow: 0 0 25px #00ffcc;
#         }
#         /* Result Box */
#         .result-box {
#             background: rgba(0,255,204,0.1);
#             padding: 25px;
#             border-radius: 15px;
#             border: 2px solid #00ffcc;
#             color: #00ffcc;
#             text-align: center;
#             font-size: 26px;
#             font-weight: bold;
#             box-shadow: 0 0 30px #00ffcc;
#             max-width: 500px;
#             margin: auto;
#         }
#         /* Flight Price Details Header */
#         .highlight-title {
#             color: #00ffcc;
#             text-align: center;
#             font-size: 32px;
#             font-weight: bold;
#             text-shadow: 0 0 10px #00ffcc, 0 0 20px #0066ff;
#             margin-bottom: 20px;
#         }
#     </style>
# """, unsafe_allow_html=True)

# # --------------------
# # LOAD MODELS
# # --------------------
# @st.cache_resource
# def load_models():
#     column_transformer = joblib.load(os.path.join("artifacts", "column_transformer.joblib"))
#     xgb_model = joblib.load(os.path.join("artifacts", "models", "xgb_flight_price_model.joblib"))
#     return column_transformer, xgb_model

# column_transformer, final_model = load_models()


# @st.cache_data  # Use cache_data for dataframes
# def load_training_data():
#     # --- IMPORTANT: CHANGE THIS PATH ---
#     # Put the path to your CSV file with the original 6000 rows
#     train_df = pd.read_csv("data/train_data.csv") 
    
#     # --- IMPORTANT: Check your column names ---
#     # Make sure these columns exist in your file
#     required_cols = ['price', 'duration', 'source', 'destination', 'total_stops']
    
#     # Basic check
#     if not all(col in train_df.columns for col in required_cols):
#         st.error("Error: The training data file is missing required columns: "
#                  "'price', 'duration', 'source', 'destination', 'total_stops'")
#         return pd.DataFrame()
        
#     # Assuming 'source' and 'destination' might be different cases, standardize them
#     train_df['source'] = train_df['source'].str.lower()
#     train_df['destination'] = train_df['destination'].str.lower()
#     return train_df

# # Load the data
# train_df = load_training_data()

# # --------------------
# # HELPER FUNCTION
# # --------------------
# def get_part_of_day_label(hour):
#     """
#     Converts a single hour integer into a part-of-day string label.
#     (This logic should match the binning in your 'part_of_day' utility)
#     """
#     if 0 <= hour < 6:
#         return "Night"
#     elif 6 <= hour < 12:
#         return "Morning"
#     elif 12 <= hour < 18:
#         return "Afternoon"
#     elif 18 <= hour <= 23:
#         return "Evening"
#     return "Unknown"

# # --------------------
# # Dropdown Options
# # --------------------
# # --------------------
# # Dropdown Options (Updated with actual dataset values)
# # --------------------
# airline_list = [
#     "Indigo", 
#     "Air India", 
#     "Jet Airways", 
#     "Spicejet", 
#     "Multiple Carriers", 
#     "Goair", 
#     "Vistara", 
#     "Air Asia", 
#     "Trujet"
# ]

# source_list = [
#     "cochin", 
#     "banglore", 
#     "hyderabad", 
#     "newdelhi", 
#     "delhi", 
#     "kolkata"
# ]

# destination_list = [
#     "delhi", 
#     "kolkata", 
#     "mumbai", 
#     "banglore", 
#     "chennai"
# ]

# additional_info_list = [
#     "no info",	
#     "in-flight meal not included",	
#     "no check-in baggage included",	
#     "1 long layover",	
#     "change airports",	
#     "business class",	
#     "1 short layover",	
#     "red-eye flight"
# ]


# # --------------------
# # TITLE
# # --------------------
# st.markdown("<h1 style='text-align: center; color: #00ffcc;'>✈️ Flight Price Prediction</h1>", unsafe_allow_html=True)
# st.markdown("<p style='text-align: center; color: white;'>Predict your flight ticket prices instantly and plan smarter.</p>", unsafe_allow_html=True)
# st.divider()

# # --------------------
# # FORM WITH GRID LAYOUT
# # --------------------
# with st.form("flight_input_form1"):
#     st.markdown("<div class='highlight-title'>📝 Flight Price Details</div>", unsafe_allow_html=True)

#     col1, col2, col3 = st.columns(3)
#     with col1:
#         airline = st.selectbox("Airline", airline_list)
#     with col2:
#         source = st.selectbox("Source", source_list)
#     with col3:
#         destination = st.selectbox("Destination", destination_list)

#     col4, col5, col6 = st.columns(3)
#     with col4:
#         duration = st.number_input("Duration (minutes)", min_value=0, placeholder="e.g. 120")
#     with col5:
#         total_stops = st.number_input("Total Stops", min_value=0, placeholder="e.g. 1")
#     with col6:
#         additional_info = st.selectbox("Additional Info", additional_info_list)

#     col7, col8 = st.columns(2)
#     with col7:
#         date_input = st.date_input("Date of Journey", value=datetime(2019, 6, 1))
#     with col8:
#         dept_time_hour = st.number_input("Departure Hour (0-23)", min_value=0, max_value=23)

#     submitted = st.form_submit_button("🔮 Predict Flight Price")

# # --------------------
# # PREDICTION + ANIMATION
# # --------------------
# if submitted:
#     dtoj_day = date_input.day
#     dtoj_month = date_input.month
#     dtoj_year = 2019

#     # --- [Your input_df creation and transformation logic... no changes here] ---
#     input_df = pd.DataFrame({
#         'airline': [airline], 'source': [source], 'destination': [destination],
#         'duration': [duration], 'total_stops': [total_stops], 'additional_info': [additional_info],
#         'dep_time_hour': [dept_time_hour], 'dtoj_day': [dtoj_day], 'dtoj_month': [dtoj_month],
#         'dtoj_year': [dtoj_year],
#     })
#     input_df = input_df.assign(
#         date=pd.to_datetime(input_df.rename(columns={
#             'dtoj_year': 'year', 'dtoj_month': 'month', 'dtoj_day': 'day'
#         })[['year', 'month', 'day']])
#     )
#     input_df = input_df.assign(is_weekend=input_df['date'].dt.weekday >= 5)
#     input_df.drop(columns=['dtoj_year'], inplace=True)
#     input_df_transformed = column_transformer.transform(input_df)
#     prediction = final_model.predict(input_df_transformed)[0]
#     # --- [End of your existing logic] ---


#     # Center result with animation
#     result_placeholder = st.empty()
#     for i in range(0, int(prediction)+1, max(1, int(prediction)//100)):
#         result_placeholder.markdown(f"<div class='result-box'>💰 Predicted Flight Price: ₹{i:,.2f}</div>", unsafe_allow_html=True)
#         time.sleep(0.01)
#     result_placeholder.markdown(f"<div class='result-box'>💰 Predicted Flight Price: ₹{prediction:,.2f}</div>", unsafe_allow_html=True)

#     # Define standard colors
#     highlight_color = "#0066ff"  # Blue for selected
#     default_color = "#00ffcc"     # Cyan for others
#     color_map = {default_color: default_color, highlight_color: highlight_color}

#     # --- [AIRLINE COMPARISON CHART (Existing) ... No changes] ---
#     st.markdown("<div class='highlight-title'>📊 Compare Airlines for Same Journey</div>", unsafe_allow_html=True)
#     airline_predictions = []
#     for air in airline_list:
#         temp_df = input_df.copy(); temp_df['airline'] = air
#         temp_transformed = column_transformer.transform(temp_df)
#         pred = final_model.predict(temp_transformed)[0]
#         airline_predictions.append((air, pred))
#     airline_df = pd.DataFrame(airline_predictions, columns=['Airline', 'Predicted Price'])
#     airline_df['Color'] = default_color
#     airline_df.loc[airline_df['Airline'] == airline, 'Color'] = highlight_color
#     fig = px.bar(
#         airline_df, x='Airline', y='Predicted Price',
#         labels={'Predicted Price': 'Predicted Price (₹)'},
#         text='Predicted Price', color='Color', color_discrete_map=color_map
#     )
#     fig.update_layout(
#         plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
#         font_color='#00ffcc', showlegend=False,
#         yaxis=dict(gridcolor='rgba(0, 255, 204, 0.2)', showgrid=True),
#         xaxis=dict(showgrid=False)
#     )
#     fig.update_traces(
#         marker_line_color='#0066ff', marker_line_width=1.5,
#         texttemplate='₹%{text:,.0f}', textposition='outside',
#         hovertemplate='<b>%{x}</b><br>Price: ₹%{y:,.2f}<extra></extra>'
#     )
#     st.plotly_chart(fig, use_container_width=True)
    
#     # --- [BEST OPTION BOX (Existing) ... No changes] ---
#     best_airline = airline_df.loc[airline_df['Predicted Price'].idxmin()]
#     st.markdown(f"""
#     <div style="
#         background: rgba(0,255,204,0.1); border: 1px solid #00ffcc; 
#         border-radius: 10px; padding: 15px; text-align: center; 
#         color: #00ccff; font-size: 18px; font-weight: bold; 
#         margin-top: 15px; box-shadow: 0 0 15px rgba(0,255,204,0.5);
#     ">
#         ✅ Best option for your journey: <b>{best_airline['Airline']}</b> at <b>₹{best_airline['Predicted Price']:,.2f}</b>
#     </div>
#     """, unsafe_allow_html=True)


#     # --- START: EXPANDED TABBED INSIGHTS SECTION ---
#     st.divider()
#     st.markdown("<div class='highlight-title'>💡 Deeper Price Insights</div>", unsafe_allow_html=True)
    
#     tab1, tab2, tab3, tab4, tab5 = st.tabs([
#         "☀️ Price by Time", 
#         "🗓️ Flexible Dates", 
#         "🚦 Price by Stops", 
#         "🗺️ Flexible Destinations",
#         "🔍 Prediction Context"
#     ])

#     # --- [Tabs 1-4 ... No changes here] ---
#     with tab1:
#         # ... [Your existing code for Tab 1] ...
#         st.markdown("<h3 style='text-align: center; color: #00ffcc;'>Check Price by Departure Time</h3>", unsafe_allow_html=True)
#         time_buckets = [(6, "Morning"), (12, "Afternoon"), (18, "Evening"), (22, "Night")]
#         time_predictions = []
#         for hour, label in time_buckets:
#             temp_df = input_df.copy(); temp_df['dep_time_hour'] = hour
#             temp_transformed = column_transformer.transform(temp_df)
#             pred = final_model.predict(temp_transformed)[0]
#             time_predictions.append({"Part of Day": label, "Predicted Price": pred})
#         day_part_df = pd.DataFrame(time_predictions)
#         user_part_of_day = get_part_of_day_label(dept_time_hour) 
#         day_part_df['Color'] = default_color
#         day_part_df.loc[day_part_df['Part of Day'] == user_part_of_day, 'Color'] = highlight_color
#         fig2 = px.bar(
#             day_part_df, x='Part of Day', y='Predicted Price', text='Predicted Price',
#             color='Color', color_discrete_map=color_map,
#             category_orders={"Part of Day": ["Morning", "Afternoon", "Evening", "Night"]}
#         )
#         fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#00ffcc', showlegend=False, yaxis=dict(gridcolor='rgba(0, 255, 204, 0.2)', title='Predicted Price (₹)'), xaxis=dict(showgrid=False, title='Time of Day'))
#         fig2.update_traces(marker_line_color='#0066ff', marker_line_width=1.5, texttemplate='₹%{text:,.0f}', textposition='outside', hovertemplate='<b>%{x}</b><br>Price: ₹%{y:,.2f}<extra></extra>')
#         st.plotly_chart(fig2, use_container_width=True)
#         st.info("Shows estimates for your selected airline and route at different times of the day.")

#     with tab2:
#         # ... [Your existing code for Tab 2] ...
#         st.markdown("<h3 style='text-align: center; color: #00ffcc;'>Check Prices for Nearby Dates</h3>", unsafe_allow_html=True)
#         date_predictions = []
#         for i in range(-3, 4):
#             new_date = date_input + pd.Timedelta(days=i)
#             temp_df = input_df.copy(); temp_df['dtoj_day'] = new_date.day; temp_df['dtoj_month'] = new_date.month; temp_df['date'] = new_date; temp_df['is_weekend'] = new_date.weekday() >= 5
#             temp_transformed = column_transformer.transform(temp_df)
#             pred = final_model.predict(temp_transformed)[0]
#             date_predictions.append({"Date": new_date.strftime("%Y-%m-%d"), "Predicted Price": pred})
#         date_df = pd.DataFrame(date_predictions)
#         fig3 = px.line(date_df, x='Date', y='Predicted Price', markers=True, text='Predicted Price')
#         fig3.update_traces(line_color=default_color, marker_color=highlight_color, texttemplate='₹%{text:,.0f}', textposition="top center", hovertemplate='<b>%{x}</b><br>Price: ₹%{y:,.2f}<extra></extra>')
#         fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#00ffcc', yaxis=dict(gridcolor='rgba(0, 255, 204, 0.2)', title='Predicted Price (₹)'), xaxis=dict(showgrid=False, title='Date'))
#         fig3.add_vline(x=date_input.strftime("%Y-%m-%d"), line_width=2, line_dash="dash", line_color="#FF4B4B")
#         st.plotly_chart(fig3, use_container_width=True)
#         st.info("Shows estimates for your flight on surrounding dates. The red dashed line is your selected date.")

#     with tab3:
#         # ... [Your existing code for Tab 3] ...
#         st.markdown("<h3 style='text-align: center; color: #00ffcc;'>Check Price by Number of Stops</h3>", unsafe_allow_html=True)
#         stops_to_check = [0, 1, 2]; stops_predictions = []
#         for s in stops_to_check:
#             temp_df = input_df.copy(); temp_df['total_stops'] = s
#             temp_transformed = column_transformer.transform(temp_df)
#             pred = final_model.predict(temp_transformed)[0]
#             stops_predictions.append({"Stops": f"{s} Stop(s)", "Predicted Price": pred})
#         stops_df = pd.DataFrame(stops_predictions)
#         user_stops_label = f"{total_stops} Stop(s)"
#         stops_df['Color'] = default_color
#         stops_df.loc[stops_df['Stops'] == user_stops_label, 'Color'] = highlight_color
#         fig4 = px.bar(stops_df, x='Stops', y='Predicted Price', text='Predicted Price', color='Color', color_discrete_map=color_map, category_orders={"Stops": ["0 Stop(s)", "1 Stop(s)", "2 Stop(s)"]})
#         fig4.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#00ffcc', showlegend=False, yaxis=dict(gridcolor='rgba(0, 255, 204, 0.2)', title='Predicted Price (₹)'), xaxis=dict(showgrid=False, title='Number of Stops'))
#         fig4.update_traces(marker_line_color='#0066ff', marker_line_width=1.5, texttemplate='₹%{text:,.0f}', textposition='outside', hovertemplate='<b>%{x}</b><br>Price: ₹%{y:,.2f}<extra></extra>')
#         st.plotly_chart(fig4, use_container_width=True)
#         st.info("Shows price estimates for your selected airline and route, but with different numbers of stops.")

#     with tab4:
#         # ... [Your existing code for Tab 4] ...
#         st.markdown("<h3 style='text-align: center; color: #00ffcc;'>Check Prices for Other Destinations</h3>", unsafe_allow_html=True)
#         destination_predictions = []
#         for dest in destination_list:
#             temp_df = input_df.copy(); temp_df['destination'] = dest
#             temp_transformed = column_transformer.transform(temp_df)
#             pred = final_model.predict(temp_transformed)[0]
#             destination_predictions.append({"Destination": dest.title(), "Predicted Price": pred})
#         dest_df = pd.DataFrame(destination_predictions)
#         user_dest_label = destination.title()
#         dest_df['Color'] = default_color
#         dest_df.loc[dest_df['Destination'] == user_dest_label, 'Color'] = highlight_color
#         fig5 = px.bar(dest_df, x='Destination', y='Predicted Price', text='Predicted Price', color='Color', color_discrete_map=color_map)
#         fig5.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#00ffcc', showlegend=False, yaxis=dict(gridcolor='rgba(0, 255, 204, 0.2)', title='Predicted Price (₹)'), xaxis=dict(showgrid=False, title='Destination City'))
#         fig5.update_traces(marker_line_color='#0066ff', marker_line_width=1.5, texttemplate='₹%{text:,.0f}', textposition='outside', hovertemplate='<b>%{x}</b><br>Price: ₹%{y:,.2f}<extra></extra>')
#         st.plotly_chart(fig5, use_container_width=True)
#         st.info("Shows estimates for flying from your source city to other destinations on the same day.")

#     # --- START: UPDATED TAB 5 CODE (CHANGES ARE HERE) ---
#     with tab5:
#         st.markdown("<h3 style='text-align: center; color: #00ffcc;'>Validating Your Prediction</h3>", unsafe_allow_html=True)
        
#         if 'train_df' not in globals() or train_df.empty:
#             st.warning("Could not load historical data for comparison.")
#         else:
#             # --- Analysis 1: Route-Specific Analysis (No Change) ---
#             st.markdown("<h4 style='text-align: center; color: #00ffcc;'>1. Analysis for Your Route</h4>", unsafe_allow_html=True)
            
#             route_data = train_df[
#                 (train_df['source'] == source.lower()) & 
#                 (train_df['destination'] == destination.lower())
#             ]
            
#             if route_data.empty:
#                 st.warning(f"No historical data found for the route: {source.title()} to {destination.title()}.")
#             else:
#                 route_data['total_stops'] = route_data['total_stops'].astype(str)
#                 fig6 = px.scatter(
#                     route_data,
#                     x='duration', y='price', color='total_stops',
#                     labels={'duration': 'Duration (minutes)', 'price': 'Price (₹)', 'total_stops': 'Total Stops'},
#                     opacity=0.7,
#                     title=f"Historical Data: {source.title()} to {destination.title()}"
#                 )
                
#                 # Bright Green Star marker
#                 fig6.add_trace(go.Scatter(
#                     x=[duration], 
#                     y=[prediction],
#                     mode='markers',
#                     marker=dict(color='#00FF00', size=15, line=dict(color='white', width=2), symbol='star'),
#                     name=f"Your Flight ({total_stops} Stop(s))",
#                     hovertemplate=f'<b>Your Flight (Predicted)</b><br>Stops: {total_stops}<br>Duration: {duration} min<br>Price: ₹{prediction:,.2f}<extra></extra>'
#                 ))
                
#                 fig6.update_layout(
#                     plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
#                     font_color='#00ffcc', showlegend=True,
#                     legend=dict(title='Total Stops', orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
#                 )
#                 st.plotly_chart(fig6, use_container_width=True)
#                 st.info("This plot shows all historical flights on your route. Your predicted flight (the green star) is plotted to see how it compares.")

#             # --- Analysis 2: Structurally-Similar Flights Analysis (NEW SCATTER PLOT) ---
#             st.markdown("<h4 style='text-align: center; color: #00ffcc; margin-top: 30px;'>2. Analysis for Similar Duration Flights (Any Route)</h4>", unsafe_allow_html=True)
            
#             duration_margin = 50 
#             duration_min = duration - duration_margin
#             duration_max = duration + duration_margin
            
#             similar_flights_data = train_df[
#                 (train_df['duration'] >= duration_min) & 
#                 (train_df['duration'] <= duration_max)
#             ]
            
#             if similar_flights_data.empty:
#                 st.warning(f"No historical data found for flights with duration between {duration_min} and {duration_max} minutes.")
#             else:
#                 # --- THIS IS THE NEW SCATTER PLOT ---
#                 # Also color by stops, as it's very insightful
#                 similar_flights_data['total_stops'] = similar_flights_data['total_stops'].astype(str)
#                 fig7 = px.scatter(
#                     similar_flights_data,
#                     x='duration',
#                     y='price',
#                     color='total_stops',
#                     labels={'duration': 'Duration (minutes)', 'price': 'Price (₹)', 'total_stops': 'Total Stops'},
#                     opacity=0.5,
#                     title=f"Price vs. Duration for Flights between {duration_min}-{duration_max} min (Any Route)"
#                 )
                
#                 # Add the green star for the user's prediction
#                 fig7.add_trace(go.Scatter(
#                     x=[duration], 
#                     y=[prediction],
#                     mode='markers',
#                     marker=dict(color='#00FF00', size=15, line=dict(color='white', width=2), symbol='star'),
#                     name=f"Your Flight ({total_stops} Stop(s))",
#                     hovertemplate=f'<b>Your Flight (Predicted)</b><br>Stops: {total_stops}<br>Duration: {duration} min<br>Price: ₹{prediction:,.2f}<extra></extra>'
#                 ))
                
#                 fig7.update_layout(
#                     plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
#                     font_color='#00ffcc', showlegend=True,
#                     legend=dict(title='Total Stops', orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
#                 )
#                 st.plotly_chart(fig7, use_container_width=True)
#                 st.info("This chart shows all historical flights (regardless of route) that have a similar duration to yours. Your flight is the green star.")
#     # --- END: UPDATED TAB 5 CODE ---


import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
import sys
import streamlit as st
import pandas as pd
import joblib
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import shap  # For SHAP
import matplotlib.pyplot as plt  # For SHAP
import numpy as np  # For SHAP
from scipy.stats import percentileofscore # For Gauge

# Add utils path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from utils.feature_utils import (
    is_north, find_part_of_month, part_of_day,
    make_month_object, remove_duration, have_info,
    duration_category
)
from utils.rbf import RBFPercentileSimilarity

# --------------------
# PAGE CONFIG
# --------------------
st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="wide"
)


# --------------------
# CUSTOM CSS
# --------------------
st.markdown("""
    <style>
        /* Black background */
        .stApp {
            background-color: black;
        }
        /* Input fields */
        .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select, .stDateInput>div>div>input {
            background: rgba(20,20,20,0.9);
            color: #00ffcc;
            border: 1px solid #00ffcc;
            border-radius: 8px;
        }
        /* Labels */
        label {
            color: #00ffcc !important;
            font-weight: bold;
        }
        /* Buttons */
        .stButton>button {
            background: linear-gradient(45deg, #00ffcc, #0066ff);
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 10px 20px;
            box-shadow: 0 0 15px #00ffcc;
            transition: transform 0.2s ease-in-out;
            border: none;
        }
        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0 0 25px #00ffcc;
        }
        /* Result Box */
        .result-box {
            background: rgba(0,255,204,0.1);
            padding: 25px;
            border-radius: 15px;
            border: 2px solid #00ffcc;
            color: #00ffcc;
            text-align: center;
            font-size: 26px;
            font-weight: bold;
            box-shadow: 0 0 30px #00ffcc;
            max-width: 500px;
            margin: auto;
        }
        
        /* --- MODIFIED: Highlight Title --- */
        .highlight-title {
            color: #00ffcc;
            text-align: center;
            font-size: 30px;
            font-weight: bold;
            padding-bottom: 10px;
            border-bottom: 2px solid rgba(0, 255, 204, 0.3); /* Subtle underline */
            margin-bottom: 25px;
        }
        
        /* --- NEW: Custom Info Box --- */
        .info-box {
            background-color: rgba(0, 102, 255, 0.1); /* Faint blue bg */
            border: 1px solid #0066ff; /* Blue border */
            border-left: 5px solid #0099FF; /* Thicker left border */
            padding: 15px;
            border-radius: 8px;
            color: #E0E0E0; /* Lighter text */
            font-family: 'sans serif';
            margin-top: 10px;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# --------------------
# LOAD MODELS
# --------------------
@st.cache_resource
def load_models():
    column_transformer = joblib.load(os.path.join("artifacts", "column_transformer.joblib"))
    xgb_model = joblib.load(os.path.join("artifacts", "models", "xgb_flight_price_model.joblib"))
    return column_transformer, xgb_model

column_transformer, final_model = load_models()


@st.cache_data  # Use cache_data for dataframes
def load_training_data():
    # --- IMPORTANT: Path to your training data ---
    data_path = "data/train_data.csv"
    
    # --- UPDATED: Added all columns needed for SHAP background and Tab 5 ---
    required_cols = [
        'price', 'duration', 'source', 'destination', 'total_stops', 'dtoj_month',
        'airline', 'additional_info', 'dep_time_hour', 'dtoj_day'
    ]
    
    try:
        train_df = pd.read_csv(data_path)
    except FileNotFoundError:
        st.error(f"Error: Training data file not found at {data_path}")
        return pd.DataFrame()

    # Basic check
    if not all(col in train_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in train_df.columns]
        st.error(f"Error: The training data file is missing required columns: {missing}")
        return pd.DataFrame()
        
    # Assuming 'source' and 'destination' might be different cases, standardize them
    train_df['source'] = train_df['source'].str.lower()
    train_df['destination'] = train_df['destination'].str.lower()
    return train_df

# Load the data
train_df = load_training_data()


# --- START: SHAP HELPER FUNCTION ---
@st.cache_resource
def load_shap_explainer(_model, _transformer):
    """
    Creates and caches a SHAP TreeExplainer.
    """
    # 1. Load the raw training data
    bg_data_raw = load_training_data()
    if bg_data_raw.empty:
        st.warning("SHAP Explainer could not be loaded as training data is missing.")
        return None

    # 2. Replicate the *exact* feature engineering from the 'if submitted' block
    if len(bg_data_raw) > 500:
        bg_sample = bg_data_raw.sample(500, random_state=42)
    else:
        bg_sample = bg_data_raw.copy()

    # --- Replicating your FE pipeline ---
    bg_sample['dtoj_year'] = 2019  # Hardcoded, just like in your prediction logic
    
    bg_sample = bg_sample.assign(
        date=pd.to_datetime(bg_sample.rename(columns={
            'dtoj_year': 'year', 
            'dtoj_month': 'month', 
            'dtoj_day': 'day'
        })[['year', 'month', 'day']])
    )
    bg_sample = bg_sample.assign(is_weekend=bg_sample['date'].dt.weekday >= 5)
    bg_sample.drop(columns=['dtoj_year', 'price'], inplace=True, errors='ignore') 
    # --- End of FE pipeline ---
    
    # 3. Transform the background data
    try:
        bg_data_transformed = _transformer.transform(bg_sample)
    except Exception as e:
        st.error(f"SHAP Error: Failed to transform background data. Mismatch in columns? Error: {e}")
        return None
    
    # 4. Create and return the explainer
    return shap.TreeExplainer(_model, bg_data_transformed)
# --- END: SHAP HELPER FUNCTION ---


# --- START: NEW GAUGE PLOT FUNCTION ---
def create_deal_gauge(price, historical_prices):
    """
    Creates a Plotly Gauge chart to show how good the deal is.
    """
    if historical_prices.empty:
        return None
        
    # Calculate percentile: "what percentage of flights are CHEAPER than this one?"
    perc = percentileofscore(historical_prices, price)
    
    if perc <= 25:
        deal_text = "Great Deal!"
        color = "#00FF00" # Green
    elif perc <= 50:
        deal_text = "Good Deal"
        color = "#ADFF2F" # Green-Yellow
    elif perc <= 75:
        deal_text = "Average Price"
        color = "#FFA500" # Orange
    else:
        deal_text = "Expensive"
        color = "#FF4B4B" # Red
        
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = price,
        number = {'prefix': "₹", 'font': {'size': 30}},
        delta = {'reference': historical_prices.mean(), 'relative': False, 'valueformat': '.0f'},
        title = {'text': f"<b>{deal_text}</b> (vs. avg. ₹{historical_prices.mean():,.0f})", 'font': {'size': 20, 'color': color}},
        gauge = {
            'axis': {'range': [historical_prices.min(), historical_prices.max()], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#666",
            'steps': [
                {'range': [historical_prices.min(), historical_prices.quantile(0.25)], 'color': 'rgba(0, 255, 0, 0.2)'},
                {'range': [historical_prices.quantile(0.25), historical_prices.quantile(0.75)], 'color': 'rgba(255, 165, 0, 0.2)'},
                {'range': [historical_prices.quantile(0.75), historical_prices.max()], 'color': 'rgba(255, 75, 75, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': price
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color="white",
        height=300
    )
    return fig
# --- END: NEW GAUGE PLOT FUNCTION ---


# --- START: NEW GLOBAL SHAP PLOT FUNCTION ---
@st.cache_resource
def generate_global_shap_plot(_explainer, _transformer):
    """
    Generates and caches the global SHAP summary plot.
    """
    if _explainer is None:
        return None
        
    try:
        # 1. Load and sample data (same as in explainer)
        bg_data_raw = load_training_data()
        if len(bg_data_raw) > 500:
            bg_sample = bg_data_raw.sample(500, random_state=42)
        else:
            bg_sample = bg_data_raw.copy()
            
        # 2. Transform data (same as in explainer)
        bg_sample_fe = bg_sample.copy()
        bg_sample_fe['dtoj_year'] = 2019
        bg_sample_fe = bg_sample_fe.assign(
            date=pd.to_datetime(bg_sample_fe.rename(columns={
                'dtoj_year': 'year', 'dtoj_month': 'month', 'dtoj_day': 'day'
            })[['year', 'month', 'day']])
        )
        bg_sample_fe = bg_sample_fe.assign(is_weekend=bg_sample_fe['date'].dt.weekday >= 5)
        bg_sample_fe.drop(columns=['dtoj_year', 'price'], inplace=True, errors='ignore')
        
        bg_transformed = _transformer.transform(bg_sample_fe)
        
        # 3. Get SHAP values for the sample
        shap_values = _explainer.shap_values(bg_transformed)
        
        # 4. Create the plot
        fig, ax = plt.subplots()
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        
        # We pass the transformed data for coloring
        shap.summary_plot(shap_values, bg_transformed, plot_type="dot", show=False)
        
        # Style for dark theme
        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_color("#00ffcc")
        ax.xaxis.label.set_color("#00ffcc")
        ax.tick_params(axis='x', colors='#00ffcc')
        ax.tick_params(axis='y', colors='#00ffcc')
        
        return fig
    except Exception as e:
        st.error(f"Error creating global SHAP plot: {e}")
        return None
# --- END: NEW GLOBAL SHAP PLOT FUNCTION ---


# --- START: LOAD SHAP EXPLAINER ---
explainer = load_shap_explainer(final_model, column_transformer)
# --- END: LOAD SHAP EXPLAINER ---


# --------------------
# HELPER FUNCTION
# --------------------
def get_part_of_day_label(hour):
    """
    Converts a single hour integer into a part-of-day string label.
    """
    if 0 <= hour < 6:
        return "Night"
    elif 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 18:
        return "Afternoon"
    elif 18 <= hour <= 23:
        return "Evening"
    return "Unknown"

# --------------------
# Dropdown Options
# --------------------
airline_list = [
    "Indigo", "Air India", "Jet Airways", "Spicejet", 
    "Multiple Carriers", "Goair", "Vistara", "Air Asia", "Trujet"
]
source_list = ["cochin", "banglore", "hyderabad", "newdelhi", "delhi", "kolkata"]
destination_list = ["delhi", "kolkata", "mumbai", "banglore", "chennai"]
additional_info_list = [
    "no info", "in-flight meal not included", "no check-in baggage included", 
    "1 long layover", "change airports", "business class", 
    "1 short layover", "red-eye flight"
]


# --------------------
# TITLE
# --------------------
st.markdown("<h1 style='text-align: center; color: #00ffcc;'>✈️ Flight Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white;'>Predict your flight ticket prices instantly and plan smarter.</p>", unsafe_allow_html=True)
st.divider()

# --------------------
# FORM WITH GRID LAYOUT
# --------------------
with st.form("flight_input_form1"):
    st.markdown("<div class='highlight-title'>📝 Flight Price Details</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        airline = st.selectbox("Airline", airline_list)
    with col2:
        source = st.selectbox("Source", source_list)
    with col3:
        destination = st.selectbox("Destination", destination_list)

    col4, col5, col6 = st.columns(3)
    with col4:
        duration = st.number_input("Duration (minutes)", min_value=0, placeholder="e.g. 120")
    with col5:
        total_stops = st.number_input("Total Stops", min_value=0, placeholder="e.g. 1")
    with col6:
        additional_info = st.selectbox("Additional Info", additional_info_list)

    col7, col8 = st.columns(2)
    with col7:
        date_input = st.date_input("Date of Journey", value=datetime(2019, 6, 1))
    with col8:
        dept_time_hour = st.number_input("Departure Hour (0-23)", min_value=0, max_value=23)

    submitted = st.form_submit_button("🔮 Predict Flight Price")


# --------------------
# PREDICTION + ANIMATION
# --------------------
if submitted:
    dtoj_day = date_input.day
    dtoj_month = date_input.month
    dtoj_year = 2019

    # --- [Your input_df creation and transformation logic... no changes here] ---
    input_df = pd.DataFrame({
        'airline': [airline], 'source': [source], 'destination': [destination],
        'duration': [duration], 'total_stops': [total_stops], 'additional_info': [additional_info],
        'dep_time_hour': [dept_time_hour], 'dtoj_day': [dtoj_day], 'dtoj_month': [dtoj_month],
        'dtoj_year': [dtoj_year],
    })
    input_df = input_df.assign(
        date=pd.to_datetime(input_df.rename(columns={
            'dtoj_year': 'year', 'dtoj_month': 'month', 'dtoj_day': 'day'
        })[['year', 'month', 'day']])
    )
    input_df = input_df.assign(is_weekend=input_df['date'].dt.weekday >= 5)
    input_df.drop(columns=['dtoj_year'], inplace=True)
    input_df_transformed = column_transformer.transform(input_df)
    prediction = final_model.predict(input_df_transformed)[0]
    # --- [End of your existing logic] ---


    # Center result with animation
    result_placeholder = st.empty()
    for i in range(0, int(prediction)+1, max(1, int(prediction)//100)):
        result_placeholder.markdown(f"<div class='result-box'>💰 Predicted Flight Price: ₹{i:,.2f}</div>", unsafe_allow_html=True)
        time.sleep(0.01)
    result_placeholder.markdown(f"<div class='result-box'>💰 Predicted Flight Price: ₹{prediction:,.2f}</div>", unsafe_allow_html=True)

    # --- START: NEW DEAL-O-METER GAUGE ---
    st.markdown("<br>", unsafe_allow_html=True) # Add a little space
    gauge_placeholder = st.empty()
    if 'train_df' not in globals() or train_df.empty:
        st.warning("Could not load historical data for deal comparison.")
    else:
        route_prices = train_df[
            (train_df['source'] == source.lower()) & 
            (train_df['destination'] == destination.lower())
        ]['price']
        
        if not route_prices.empty:
            gauge_fig = create_deal_gauge(prediction, route_prices)
            gauge_placeholder.plotly_chart(gauge_fig, use_container_width=True)
    # --- END: NEW DEAL-O-METER GAUGE ---


    # Define standard colors
    highlight_color = "#0099FF"  # A richer blue
    default_color = "#00BFA6"     # A darker, richer cyan
    color_map = {default_color: default_color, highlight_color: highlight_color}

    # --- [AIRLINE COMPARISON CHART (Existing) ... No changes] ---
    st.markdown("<div class='highlight-title'>📊 Compare Airlines for Same Journey</div>", unsafe_allow_html=True)
    airline_predictions = []
    for air in airline_list:
        temp_df = input_df.copy(); temp_df['airline'] = air
        temp_transformed = column_transformer.transform(temp_df)
        pred = final_model.predict(temp_transformed)[0]
        airline_predictions.append((air, pred))
    airline_df = pd.DataFrame(airline_predictions, columns=['Airline', 'Predicted Price'])
    airline_df['Color'] = default_color
    airline_df.loc[airline_df['Airline'] == airline, 'Color'] = highlight_color
    fig = px.bar(
        airline_df, x='Airline', y='Predicted Price',
        labels={'Predicted Price': 'Predicted Price (₹)'},
        text='Predicted Price', color='Color', color_discrete_map=color_map
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_color='#00ffcc', showlegend=False,
        yaxis=dict(gridcolor='rgba(0, 255, 204, 0.2)', showgrid=True),
        xaxis=dict(showgrid=False),
        bargap=0.3  # --- ADDED: Creates space between bars
    )
    fig.update_traces(
        marker_line_color=highlight_color, marker_line_width=1.5,
        texttemplate='₹%{text:,.0f}', textposition='outside',
        hovertemplate='<b>%{x}</b><br>Price: ₹%{y:,.2f}<extra></extra>'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # --- [BEST OPTION BOX (Existing) ... No changes] ---
    best_airline = airline_df.loc[airline_df['Predicted Price'].idxmin()]
    st.markdown(f"""
    <div style="
        background: rgba(0,255,204,0.1); border: 1px solid #00ffcc; 
        border-radius: 10px; padding: 15px; text-align: center; 
        color: #00ccff; font-size: 18px; font-weight: bold; 
        margin-top: 15px; box-shadow: 0 0 15px rgba(0,255,204,0.5);
    ">
        ✅ Best option for your journey: <b>{best_airline['Airline']}</b> at <b>₹{best_airline['Predicted Price']:,.2f}</b>
    </div>
    """, unsafe_allow_html=True)


    # --- START: 5-TAB INSIGHTS SECTION ---
    st.divider()
    st.markdown("<div class='highlight-title'>💡 Deeper Price Insights</div>", unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "☀️ Price by Time", 
        "🗓️ Flexible Dates", 
        "🚦 Price by Stops", 
        "🗺️ Flexible Sources",
        "🔍 Prediction Context"
    ])

    # --- [Tabs 1-4 with .info-box] ---
    with tab1:
        st.markdown("<h3 style='text-align: center; color: #00ffcc;'>Check Price by Departure Time</h3>", unsafe_allow_html=True)
        time_buckets = [(6, "Morning"), (12, "Afternoon"), (18, "Evening"), (22, "Night")]
        time_predictions = []
        for hour, label in time_buckets:
            temp_df = input_df.copy(); temp_df['dep_time_hour'] = hour
            temp_transformed = column_transformer.transform(temp_df)
            pred = final_model.predict(temp_transformed)[0]
            time_predictions.append({"Part of Day": label, "Predicted Price": pred})
        day_part_df = pd.DataFrame(time_predictions)
        user_part_of_day = get_part_of_day_label(dept_time_hour) 
        day_part_df['Color'] = default_color
        day_part_df.loc[day_part_df['Part of Day'] == user_part_of_day, 'Color'] = highlight_color
        fig2 = px.bar(
            day_part_df, x='Part of Day', y='Predicted Price', text='Predicted Price',
            color='Color', color_discrete_map=color_map,
            category_orders={"Part of Day": ["Morning", "Afternoon", "Evening", "Night"]}
        )
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
            font_color='#00ffcc', showlegend=False, 
            yaxis=dict(gridcolor='rgba(0, 255, 204, 0.2)', title='Predicted Price (₹)'), 
            xaxis=dict(showgrid=False, title='Time of Day'),
            bargap=0.3
        )
        fig2.update_traces(marker_line_color=highlight_color, marker_line_width=1.5, texttemplate='₹%{text:,.0f}', textposition='outside', hovertemplate='<b>%{x}</b><br>Price: ₹%{y:,.2f}<extra></extra>')
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("<div class='info-box'>Shows estimates for your selected airline and route at different times of the day.</div>", unsafe_allow_html=True)


    with tab2:
        st.markdown("<h3 style='text-align: center; color: #00ffcc;'>Check Prices for Nearby Dates</h3>", unsafe_allow_html=True)
        date_predictions = []
        for i in range(-3, 4):
            new_date = date_input + pd.Timedelta(days=i)
            temp_df = input_df.copy(); temp_df['dtoj_day'] = new_date.day; temp_df['dtoj_month'] = new_date.month; temp_df['date'] = new_date; temp_df['is_weekend'] = new_date.weekday() >= 5
            temp_transformed = column_transformer.transform(temp_df)
            pred = final_model.predict(temp_transformed)[0]
            date_predictions.append({"Date": new_date.strftime("%Y-%m-%d"), "Predicted Price": pred})
        date_df = pd.DataFrame(date_predictions)
        fig3 = px.line(date_df, x='Date', y='Predicted Price', markers=True, text='Predicted Price')
        fig3.update_traces(line_color=default_color, marker_color=highlight_color, texttemplate='₹%{text:,.0f}', textposition="top center", hovertemplate='<b>%{x}</b><br>Price: ₹%{y:,.2f}<extra></extra>')
        fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#00ffcc', yaxis=dict(gridcolor='rgba(0, 255, 204, 0.2)', title='Predicted Price (₹)'), xaxis=dict(showgrid=False, title='Date'))
        fig3.add_vline(x=date_input.strftime("%Y-%m-%d"), line_width=2, line_dash="dash", line_color="#FF4B4B")
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("<div class='info-box'>Shows estimates for your flight on surrounding dates. The red dashed line is your selected date.</div>", unsafe_allow_html=True)


    with tab3:
        st.markdown("<h3 style='text-align: center; color: #00ffcc;'>Check Price by Number of Stops</h3>", unsafe_allow_html=True)
        stops_to_check = [0, 1, 2]; stops_predictions = []
        for s in stops_to_check:
            temp_df = input_df.copy(); temp_df['total_stops'] = s
            temp_transformed = column_transformer.transform(temp_df)
            pred = final_model.predict(temp_transformed)[0]
            stops_predictions.append({"Stops": f"{s} Stop(s)", "Predicted Price": pred})
        stops_df = pd.DataFrame(stops_predictions)
        user_stops_label = f"{total_stops} Stop(s)"
        stops_df['Color'] = default_color
        stops_df.loc[stops_df['Stops'] == user_stops_label, 'Color'] = highlight_color
        fig4 = px.bar(stops_df, x='Stops', y='Predicted Price', text='Predicted Price', color='Color', color_discrete_map=color_map, category_orders={"Stops": ["0 Stop(s)", "1 Stop(s)", "2 Stop(s)"]})
        fig4.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
            font_color='#00ffcc', showlegend=False, 
            yaxis=dict(gridcolor='rgba(0, 255, 204, 0.2)', title='Predicted Price (₹)'), 
            xaxis=dict(showgrid=False, title='Number of Stops'),
            bargap=0.3
        )
        fig4.update_traces(marker_line_color=highlight_color, marker_line_width=1.5, texttemplate='₹%{text:,.0f}', textposition='outside', hovertemplate='<b>%{x}</b><br>Price: ₹%{y:,.2f}<extra></extra>')
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("<div class='info-box'>Shows price estimates for your selected airline and route, but with different numbers of stops.</div>", unsafe_allow_html=True)

# --- START: UPDATED TAB 4 (FLEXIBLE SOURCES) ---
    with tab4:
        st.markdown("<h3 style='text-align: center; color: #00ffcc;'>Check Prices from Other Sources</h3>", unsafe_allow_html=True)

        source_predictions = []

        # Use the source_list you defined at the top of your script
        for src in source_list:
            temp_df = input_df.copy()
            temp_df['source'] = src # <-- Change the source

            temp_transformed = column_transformer.transform(temp_df)
            pred = final_model.predict(temp_transformed)[0]
            source_predictions.append({"Source": src.title(), "Predicted Price": pred})

        source_df = pd.DataFrame(source_predictions, columns=['Source', 'Predicted Price'])

        # Highlight the user's original selection
        user_source_label = source.title() # 'source' is from the form
        source_df['Color'] = default_color
        source_df.loc[source_df['Source'] == user_source_label, 'Color'] = highlight_color

        fig5 = px.bar(
            source_df, x='Source', y='Predicted Price', text='Predicted Price',
            color='Color', color_discrete_map=color_map
        )
        fig5.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font_color='#00ffcc', showlegend=False,
            yaxis=dict(gridcolor='rgba(0, 255, 204, 0.2)', title='Predicted Price (₹)'),
            xaxis=dict(showgrid=False, title='Source City'), # <-- Changed label
            bargap=0.3
        )
        fig5.update_traces(marker_line_color=highlight_color, marker_line_width=1.5, texttemplate='₹%{text:,.0f}', textposition='outside', hovertemplate='<b>%{x}</b><br>Price: ₹%{y:,.2f}<extra></extra>')
        st.plotly_chart(fig5, use_container_width=True)
        # --- Changed info text ---
        st.markdown("<div class='info-box'>Shows estimates for flying to your chosen destination from other source cities on the same day.</div>", unsafe_allow_html=True)
    # --- END: UPDATED TAB 4 ---

    with tab5:
        # ... [Your existing code for Tab 5, including the month toggle] ...
        st.markdown("<h3 style='text-align: center; color: #00ffcc;'>Validating Your Prediction</h3>", unsafe_allow_html=True)
        if 'train_df' not in globals() or train_df.empty:
            st.warning("Could not load historical data for comparison.")
        else:
            # --- Analysis 1: Route-Specific Analysis (WITH MONTH TOGGLE) ---
            st.markdown("<h4 style='text-align: center; color: #00ffcc;'>1. Analysis for Your Route</h4>", unsafe_allow_html=True)
            user_month = date_input.month
            month_name = date_input.strftime("%B")
            filter_by_month = st.toggle(f"Show historical data for {month_name} only", value=False)
            route_data = train_df[(train_df['source'] == source.lower()) & (train_df['destination'] == destination.lower())]
            if filter_by_month:
                route_data = route_data[route_data['dtoj_month'] == user_month]
                plot_title = f"Historical Data: {source.title()} to {destination.title()} (in {month_name})"
            else:
                plot_title = f"Historical Data: {source.title()} to {destination.title()} (All Months)"
            if route_data.empty:
                if filter_by_month: st.warning(f"No historical data found for this route in {month_name}.")
                else: st.warning(f"No historical data found for the route: {source.title()} to {destination.title()}.")
            else:
                route_data['total_stops'] = route_data['total_stops'].astype(str)
                fig6 = px.scatter(
                    route_data, x='duration', y='price', color='total_stops',
                    labels={'duration': 'Duration (minutes)', 'price': 'Price (₹)', 'total_stops': 'Total Stops'},
                    opacity=0.7, title=plot_title
                )
                fig6.add_trace(go.Scatter(
                    x=[duration], y=[prediction], mode='markers',
                    marker=dict(color='#00FF00', size=15, line=dict(color='white', width=2), symbol='star'),
                    name=f"Your Flight ({total_stops} Stop(s))",
                    hovertemplate=f'<b>Your Flight (Predicted)</b><br>Stops: {total_stops}<br>Duration: {duration} min<br>Price: ₹{prediction:,.2f}<extra></extra>'
                ))
                fig6.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#00ffcc', showlegend=True, legend=dict(title='Total Stops', orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig6, use_container_width=True)
                st.markdown("<div class='info-box'>This plot shows all historical flights on your route. Your predicted flight (the green star) is plotted to see how it compares. Use the toggle to see data for a specific month.</div>", unsafe_allow_html=True)


            # --- Analysis 2: Structurally-Similar Flights Analysis (No Change) ---
            st.markdown("<h4 style='text-align: center; color: #00ffcc; margin-top: 30px;'>2. Analysis for Similar Duration Flights (Any Route)</h4>", unsafe_allow_html=True)
            duration_margin = 50 
            duration_min = duration - duration_margin
            duration_max = duration + duration_margin
            similar_flights_data = train_df[(train_df['duration'] >= duration_min) & (train_df['duration'] <= duration_max)]
            if similar_flights_data.empty:
                st.warning(f"No historical data found for flights with duration between {duration_min} and {duration_max} minutes.")
            else:
                similar_flights_data['total_stops'] = similar_flights_data['total_stops'].astype(str)
                fig7 = px.scatter(
                    similar_flights_data, x='duration', y='price', color='total_stops',
                    labels={'duration': 'Duration (minutes)', 'price': 'Price (₹)', 'total_stops': 'Total Stops'},
                    opacity=0.5,
                    title=f"Price vs. Duration for Flights between {duration_min}-{duration_max} min (Any Route)"
                )
                fig7.add_trace(go.Scatter(
                    x=[duration], y=[prediction], mode='markers',
                    marker=dict(color='#00FF00', size=15, line=dict(color='white', width=2), symbol='star'),
                    name=f"Your Flight ({total_stops} Stop(s))",
                    hovertemplate=f'<b>Your Flight (Predicted)</b><br>Stops: {total_stops}<br>Duration: {duration} min<br>Price: ₹{prediction:,.2f}<extra></extra>'
                ))
                fig7.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#00ffcc', showlegend=True, legend=dict(title='Total Stops', orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig7, use_container_width=True)
                st.markdown("<div class='info-box'>This chart shows all historical flights (regardless of route) that have a similar duration to yours. Your flight is the green star.</div>", unsafe_allow_html=True)
    # --- END: 5-TAB INSIGHTS SECTION ---


    # --- START: SHAP WATERFALL PLOT SECTION (No Change) ---
    st.divider()
    st.markdown("<div class='highlight-title'>💲 Price Breakdown (Why This Price?)</div>", unsafe_allow_html=True)
    
    if 'explainer' not in globals() or explainer is None:
        st.warning("The SHAP Explainer is not available. Could not generate price breakdown.")
    else:
        try:
            shap_values = explainer.shap_values(input_df_transformed)
            
            if hasattr(input_df_transformed, "toarray"):
                input_values_dense = input_df_transformed.toarray()[0]
            else:
                input_values_dense = input_df_transformed[0]

            shap_explanation = shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=input_values_dense
            )

            st.markdown("<h4 style='text-align: center; color: #00ffcc;'>Price Contribution by Feature</h4>", unsafe_allow_html=True)
            
            fig, ax = plt.subplots()
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            
            shap.waterfall_plot(shap_explanation, max_display=10, show=False)
            
            for text in ax.get_xticklabels() + ax.get_yticklabels():
                text.set_color("#00ffcc")
            ax.xaxis.label.set_color("#00ffcc")
            ax.yaxis.label.set_color("#00ffcc")
            
            st.pyplot(fig, bbox_inches='tight')
            plt.close(fig) # Clear the figure
            
            info_text = (
                "This chart shows how the model's features contributed to the final price. "
                f"The model's base price (average) is ₹{explainer.expected_value:,.2f}. "
                "Red bars pushed the price up, and blue bars pushed it down. "
                "(Note: Feature names are shown as indices like 'Feature 0' because of a custom transformer in the model pipeline.)"
            )
            st.markdown(f"<div class='info-box'>{info_text}</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred while generating the SHAP plot: {e}")
    # --- END: SHAP WATERFALL PLOT SECTION ---


    # --- START: NEW GLOBAL SHAP SUMMARY PLOT ---
    st.divider()
    st.markdown("<div class='highlight-title'>🌎 What Matters Most for All Flights?</div>", unsafe_allow_html=True)

    if 'explainer' not in globals() or explainer is None:
        st.warning("The SHAP Explainer is not available. Could not generate the global summary plot.")
    else:
        # Call the new helper function
        fig_summary = generate_global_shap_plot(explainer, column_transformer)
        if fig_summary:
            st.pyplot(fig_summary, bbox_inches='tight')
            plt.close(fig_summary) # Clear the figure
            
            summary_info = (
                "This plot shows the most important features for the model *overall*.<br><br>"
                "<b>Rows:</b> Features are ranked by importance (top is most important).<br>"
                "<b>Dots:</b> Each dot is a flight from a test sample.<br>"
                "<b>Color:</b> Red dots mean the feature had a *high value* (e.g., long duration), blue means a *low value*.<br>"
                "<b>X-Axis:</b> Dots on the right pushed the price *up*, dots on the left pushed the price *down*.<br><br>"
                "<b>Example:</b> A row with red dots on the right and blue dots on the left proves the model learned a logical rule (e.g., 'high duration = high price')."
            )
            st.markdown(f"<div class='info-box'>{summary_info}</div>", unsafe_allow_html=True)
    # --- END: NEW GLOBAL SHAP SUMMARY PLOT ---