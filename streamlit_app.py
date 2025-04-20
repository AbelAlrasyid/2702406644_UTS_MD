# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gzip

# Load model dan scaler
@st.cache_resource
def load_model_and_scaler():
    with gzip.open("best_xgb_model.pkl.gz", "rb") as f:
        model = pickle.load(f)
    with open("standard_scaler.pkl", "rb") as f:
        standard_scaler = pickle.load(f)
    with open("robust_scaler.pkl", "rb") as f:
        robust_scaler = pickle.load(f)
    with open("columns.pkl", "rb") as f:
        columns = pickle.load(f)
    return model, standard_scaler, robust_scaler, columns

model, standard_scaler, robust_scaler, columns = load_model_and_scaler()

# Kolom untuk scaler
standard_cols = ['arrival_month', 'arrival_date']
robust_cols = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights',
               'no_of_week_nights', 'required_car_parking_space', 'lead_time',
               'arrival_year', 'repeated_guest', 'no_of_previous_cancellations',
               'no_of_previous_bookings_not_canceled', 'avg_price_per_room',
               'no_of_special_requests']

# Form input
st.title("Hotel Booking Cancellation Prediction")

with st.form("input_form"):
    no_of_adults = st.number_input("Number of Adults", min_value=0, value=2)
    no_of_children = st.number_input("Number of Children", min_value=0, value=0)
    no_of_weekend_nights = st.number_input("Weekend Nights", min_value=0, value=1)
    no_of_week_nights = st.number_input("Week Nights", min_value=0, value=2)
    required_car_parking_space = st.selectbox("Car Parking Required?", [0, 1])
    lead_time = st.number_input("Lead Time", min_value=0, value=60)
    arrival_year = st.selectbox("Arrival Year", [2017, 2018])
    arrival_month = st.selectbox("Arrival Month", list(range(1, 13)))
    arrival_date = st.selectbox("Arrival Date", list(range(1, 32)))
    repeated_guest = st.selectbox("Repeated Guest?", [0, 1])
    no_of_previous_cancellations = st.number_input("Previous Cancellations", min_value=0, value=0)
    no_of_previous_bookings_not_canceled = st.number_input("Previous Non-Canceled Bookings", min_value=0, value=0)
    avg_price_per_room = st.number_input("Average Price per Room", min_value=0.0, value=100.0)
    no_of_special_requests = st.number_input("Special Requests", min_value=0, value=0)

    type_of_meal_plan = st.selectbox("Meal Plan", ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'])
    room_type_reserved = st.selectbox("Room Type", ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7'])
    market_segment_type = st.selectbox("Market Segment", ['Online', 'Offline', 'Corporate', 'Aviation', 'Complementary'])

    submitted = st.form_submit_button("Predict")

if submitted:
    # Buat DataFrame
    input_dict = {
        'no_of_adults': no_of_adults,
        'no_of_children': no_of_children,
        'no_of_weekend_nights': no_of_weekend_nights,
        'no_of_week_nights': no_of_week_nights,
        'required_car_parking_space': required_car_parking_space,
        'lead_time': lead_time,
        'arrival_year': arrival_year,
        'arrival_month': arrival_month,
        'arrival_date': arrival_date,
        'repeated_guest': repeated_guest,
        'no_of_previous_cancellations': no_of_previous_cancellations,
        'no_of_previous_bookings_not_canceled': no_of_previous_bookings_not_canceled,
        'avg_price_per_room': avg_price_per_room,
        'no_of_special_requests': no_of_special_requests,
        'type_of_meal_plan': type_of_meal_plan,
        'room_type_reserved': room_type_reserved,
        'market_segment_type': market_segment_type
    }

    input_df = pd.DataFrame([input_dict])

    # One-hot encoding
    input_df = pd.get_dummies(input_df, columns=['type_of_meal_plan', 'room_type_reserved', 'market_segment_type'])
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # Scaling
    input_df[standard_cols] = standard_scaler.transform(input_df[standard_cols])
    input_df[robust_cols] = robust_scaler.transform(input_df[robust_cols])

    # Prediction
    prediction = model.predict(input_df)[0]
    label = 'Canceled' if prediction == 1 else 'Not Canceled'

    st.subheader("Prediction Result")
    st.success(f"The booking is predicted to be: **{label}**")
