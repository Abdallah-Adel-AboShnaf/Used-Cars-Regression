import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

# 1. Page Configuration
st.set_page_config(page_title="Used Car Price Predictor", layout="wide")

st.title("🚗 Used Car Price Predictor")
st.markdown("""
This app predicts the price of a used car based on its specifications.
It replicates the **Data Cleaning** and **Feature Engineering** pipeline from the notebook
and uses a **Decision Tree Regressor** for prediction.
""")

#  GLOBAL HELPER FUNCTIONS
# Defined globally to avoid PickleError during Streamlit caching
def engineering_pipeline(data):
    """
    Applies feature engineering transformations to the dataframe.
    """
    data = data.copy()

    # Helper to detect turbo safely
    def has_turbo(text):
        if pd.isna(text): return 0
        return int("turbo" in str(text).lower())

    # Feature: Is Turbo
    if "engine" in data.columns:
        data["is_turbo"] = data["engine"].apply(has_turbo)
    else:
        data["is_turbo"] = 0

    # Feature: Is Automatic
    if "transmission" in data.columns:
        data["is_automatic"] = data["transmission"].str.contains("Auto", case=False, na=False).astype(int)
    else:
        data["is_automatic"] = 0

    # Feature: Drivetrain types
    if "drivetrain" in data.columns:
        data["is_AWD"] = data["drivetrain"].str.contains("AWD", case=False, na=False).astype(int)
        data["is_FWD"] = data["drivetrain"].str.contains("FWD", case=False, na=False).astype(int)
        data["is_RWD"] = data["drivetrain"].str.contains("RWD", case=False, na=False).astype(int)
    else:
        data["is_AWD"] = 0
        data["is_FWD"] = 0
        data["is_RWD"] = 0

    # Feature: Total Usage Score
    if "one_owner" in data.columns and "personal_use_only" in data.columns:
        data["total_usage_score"] = data[["one_owner", "personal_use_only"]].sum(axis=1)
    else:
        data["total_usage_score"] = 0

    return data

# --- CACHED DATA LOADING
@st.cache_data
def load_and_prep_data():
    """
    Loads data, cleans it, performs encoding, and trains the model.
    Returns the model, scalers, encoders, and the cleaned dataframe structure.
    """
    try:
        # Load Data
        #df = pd.read_csv('data/cars.csv')
        DATA_URL = "https://drive.google.com/uc?id=1sZKbkx6u0_cn1ESKe0GzUdPugA-dCaYg"

        @st.cache_data
        def load_data():
            return pd.read_csv(DATA_URL)

        df = load_data()

        # --- Data Cleaning ---
        # Drop columns as per notebook analysis
        cols_to_drop = ['price_drop', 'seller_rating']
        df.drop([c for c in cols_to_drop if c in df.columns], axis=1, inplace=True)
        df.drop_duplicates(inplace=True)

        # --- Handling Missing Values ---
        target = 'price'
        features_df = df.drop(columns=[target])

        cols_numerical = features_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cols_categorical = features_df.select_dtypes(include=['object']).columns.tolist()

        # Impute Numerical with Median
        imputer_num = SimpleImputer(strategy='median')
        df[cols_numerical] = imputer_num.fit_transform(df[cols_numerical])

        # Impute Categorical with Most Frequent
        imputer_cat = SimpleImputer(strategy='most_frequent')
        df[cols_categorical] = imputer_cat.fit_transform(df[cols_categorical])

        # --- Feature Engineering ---
        # Call global function
        df = engineering_pipeline(df)

        # --- Preparation for Training ---
        X = df.drop(columns=[target])
        y = df[target]

        cols_categorical_final = X.select_dtypes(include=['object']).columns.tolist()

        # --- Encoding (Label Encoding) ---
        encoders = {}
        for col in cols_categorical_final:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le

        # --- Feature Scaling ---
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # --- Model Training ---
        model = DecisionTreeRegressor(random_state=42)
        model.fit(X_scaled, y)

        return model, scaler, encoders, df

    except Exception as e:
        # Return error message as the last element if it fails
        return None, None, None, str(e)

# --- MAIN APP EXECUTION ---

# Load artifacts
model, scaler, encoders, raw_df = load_and_prep_data()

# Check if loading failed (raw_df would be an error string)
if isinstance(raw_df, str):
    st.error(f"Error loading data: {raw_df}")
    st.info("Please ensure 'data/cars.csv' exists in the correct directory.")

else:
    # --- Sidebar: User Input Form ---
    st.sidebar.header("📝 Car Configuration")

    def user_input_features():
        inputs = {}

        # Helper to get sorted unique values for dropdowns
        def get_unique(col):
            if col in raw_df.columns:
                return sorted(raw_df[col].astype(str).unique())
            return []

        st.sidebar.subheader("Basic Specs")
        inputs['manufacturer'] = st.sidebar.selectbox('Manufacturer', get_unique('manufacturer'))

        # Filter models based on manufacturer
        if 'manufacturer' in raw_df.columns and 'model' in raw_df.columns:
            filtered_models = raw_df[raw_df['manufacturer'] == inputs['manufacturer']]['model'].unique()
            inputs['model'] = st.sidebar.selectbox('Model', sorted(filtered_models))
        else:
            inputs['model'] = st.sidebar.text_input('Model', 'Any')

        inputs['year'] = st.sidebar.number_input('Year', min_value=1990, max_value=2025, value=2019)
        inputs['mileage'] = st.sidebar.number_input('Mileage', min_value=0, value=50000)

        st.sidebar.subheader("Technical Specs")
        inputs['engine'] = st.sidebar.selectbox('Engine', get_unique('engine'))
        inputs['transmission'] = st.sidebar.selectbox('Transmission', get_unique('transmission'))
        inputs['drivetrain'] = st.sidebar.selectbox('Drivetrain', get_unique('drivetrain'))
        inputs['fuel_type'] = st.sidebar.selectbox('Fuel Type', get_unique('fuel_type'))
        inputs['mpg'] = st.sidebar.selectbox('MPG', get_unique('mpg'))

        st.sidebar.subheader("Appearance & History")
        inputs['exterior_color'] = st.sidebar.selectbox('Exterior Color', get_unique('exterior_color'))
        inputs['interior_color'] = st.sidebar.selectbox('Interior Color', get_unique('interior_color'))
        inputs['seller_name'] = st.sidebar.selectbox('Seller Name', get_unique('seller_name'))

        # Binary features
        inputs['accidents_or_damage'] = st.sidebar.radio("Accidents?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        inputs['one_owner'] = st.sidebar.radio("One Owner?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        inputs['personal_use_only'] = st.sidebar.radio("Personal Use?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")

        inputs['driver_rating'] = st.sidebar.slider("Driver Rating", 0.0, 5.0, 4.5)
        inputs['driver_reviews_num'] = st.sidebar.number_input("Number of Reviews", min_value=0, value=50)

        return pd.DataFrame([inputs])

    input_df = user_input_features()

    # --- Main Panel: Prediction Logic ---
    st.subheader("Selected Details")
    st.dataframe(input_df)

    if st.button("Predict Price"):
        # 1. Apply Feature Engineering
        processed_input = engineering_pipeline(input_df)

        try:
            # 2. Apply Label Encoding
            for col, le in encoders.items():
                if col in processed_input.columns:
                    val = str(processed_input[col].iloc[0])
                    # Handle unseen labels
                    if val in le.classes_:
                        processed_input[col] = le.transform([val])
                    else:
                        # Fallback for unseen values (default to 0 or mode)
                        processed_input[col] = 0

            # 3. Align Columns with Scaler
            # If scaler has feature names saved, ensure alignment
            if hasattr(scaler, 'feature_names_in_'):
                for col in scaler.feature_names_in_:
                    if col not in processed_input.columns:
                        processed_input[col] = 0
                # Reorder columns to match training data
                processed_input = processed_input[scaler.feature_names_in_]

            # 4. Scaling
            input_scaled = scaler.transform(processed_input)

            # 5. Prediction
            prediction = model.predict(input_scaled)

            st.success(f"### Estimated Price: ${prediction[0]:,.2f}")

        except Exception as e:
            st.error("An error occurred during prediction.")
            st.warning(f"Detail: {e}")