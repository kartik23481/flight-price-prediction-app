# AeroPrice Insight ✈️💰

This project provides an end-to-end solution for predicting flight ticket prices using machine learning. It features a complete pipeline including data preprocessing, feature engineering, model training (XGBoost), and a user-friendly interactive web application built with Streamlit.

The application not only predicts the price for a given flight configuration but also provides several actionable insights to help users find the best deals by exploring trade-offs like airline choice, departure time, travel dates, number of stops, and departure city.

[![App Interface - Click to Watch Demo](interface.png)](https://www.youtube.com/watch?v=t00O_rV1O5I)
<br>*Click the image above to watch a video demonstration of the app.*

## ✨ Features

* **Accurate Price Prediction:** Enter flight details (airline, route, date, time, stops, duration) and get an instant price prediction.
* **Deal-o-Meter Gauge:** Instantly see if the predicted price is a "Great Deal," "Good Deal," "Average," or "Expensive" compared to historical prices for that specific route (if available).
* **Airline Comparison:** View predicted prices for the *same journey* across all available airlines, highlighting the user's selected airline.
* **Deeper Insights (via Tabs):**
    * **Price by Time:** See how the price changes for morning, afternoon, evening, or night departures.
    * **Flexible Dates:** Explore price trends using a line chart for a few days before and after your selected date.
    * **Price by Stops:** Compare the cost of non-stop vs. 1-stop or 2-stop flights for the same route and airline.
    * **Flexible Sources:** Check prices for flying *to* your chosen destination *from* different origin cities on the same day.
    * **Prediction Context:** Validate the prediction against historical data:
        * View similar flights on the *same route*, filterable by month.
        * View similar flights based on *duration* across *any route*.
* **Price Breakdown (SHAP Waterfall Plot):** Understand *why* the price is what it is by seeing how each feature contributed positively or negatively, starting from the average price.
* **Global Model Insights (SHAP Summary Plot):** See which factors (like duration, airline, stops) matter most for flight prices *overall*, demonstrating the model's logic.

## 🛠️ Technology Stack

* **Language:** Python 3
* **Data Science:** Pandas, NumPy, Scikit-learn, SciPy
* **Machine Learning:** XGBoost
* **Web Framework:** Streamlit
* **Visualization:** Plotly, Matplotlib, SHAP
* **Model Persistence:** Joblib

## 📂 Project Structure

* **`artifacts/`**: Contains saved machine learning objects.
    * `column_transformer.joblib`: The saved preprocessing pipeline.
    * **`models/`**: Contains the trained machine learning models.
        * `xgb_flight_price_model.joblib`: The saved XGBoost regression model.
* **`data/`**: Holds the datasets used.
    * `train_data.csv`: Training data (also used for context plots in the app).
* **`notebooks/`**: Jupyter notebooks for experimentation and development.
    * `exploratory_data_analysis.ipynb`: Notebook for initial data exploration.
    * `feature_engineering.ipynb`: Notebook detailing feature creation.
    * `model_training_tuning.ipynb`: Notebook for training and tuning the model.
    * *(Include others as needed)*
* **`utils/`**: Helper Python modules with reusable functions.
    * `feature_utils.py`: Functions for feature engineering.
    * `rbf.py`: Functions related to RBF (if applicable).
* `app.py`: The main script for the Streamlit web application.
* `requirements.txt`: Lists the Python packages required for the project.
* `.gitignore`: Specifies intentionally untracked files that Git should ignore.
* `README.md`: This file, providing information about the project.

*(Adjust structure if yours differs)*

## 🚀 Setup and Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/kartik23481/flight-price-prediction-app.git](https://github.com/kartik23481/flight-price-prediction-app.git)
    cd flight-price-prediction-app
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r freezed_new_flight_venv_requirements.txt
    ```

4.  **Ensure Data and Artifacts:**
    Make sure the training data file (`data/train_data.csv`) and the saved model/transformer (`artifacts/`) are present in the correct locations.

5.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
    The application should open automatically in your web browser.

## 📈 Model Performance

The XGBoost regression model achieves an **R² score of approximately 86%** on the validation set, indicating a reasonably good fit for predicting flight prices based on the available features and data size (approx. 6000 rows).

## 🔮 Future Improvements

* Modify the original model training script to correctly implement `get_feature_names_out` for custom transformers in the Scikit-learn pipeline. Re-save the `column_transformer.joblib` to enable user-friendly feature names in SHAP plots.
* Implement a complete **CI/CD Pipeline:** Set up automated testing (e.g., using GitHub Actions) and continuous deployment to a cloud platform.
* Integrate an **Interactive AI Chatbot:** Add an agentic AI chatbot (using Microsoft Autogen Framework) that allows users to ask natural language questions about the training data, generating insights and visualizations on the fly. 🤖📊
* Deploy the Streamlit app to a cloud platform (e.g., Streamlit Community Cloud, Heroku, AWS) for public access.
* Incorporate more recent or a larger dataset for potentially improved model accuracy and robustness.
* Experiment with alternative regression models (e.g., LightGBM, CatBoost) or further hyperparameter tuning.
* Add enhanced error handling for invalid user inputs (e.g., duration incompatible with stops, impossible routes).