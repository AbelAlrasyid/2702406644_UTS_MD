import streamlit as st
import pandas as pd
import pickle
import datetime  # <-- Tambahan

# Load model dan scaler
@st.cache_resource
def load_all():
    with open("best_xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("standard_scaler.pkl", "rb") as f:
        standard_scaler = pickle.load(f)
    with open("robust_scaler.pkl", "rb") as f:
        robust_scaler = pickle.load(f)
    with open("columns.pkl", "rb") as f:
        all_columns = pickle.load(f)
    return model, standard_scaler, robust_scaler, all_columns

model, standard_scaler, robust_scaler, all_columns = load_all()

cat_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
standard_cols = ['arrival_month', 'arrival_date']
robust_cols = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights',
               'no_of_week_nights', 'required_car_parking_space', 'lead_time',
               'arrival_year', 'repeated_guest', 'no_of_previous_cancellations',
               'no_of_previous_bookings_not_canceled', 'avg_price_per_room',
               'no_of_special_requests']

def preprocess_input(data):
    df = pd.DataFrame([data])
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    df = df.reindex(columns=all_columns, fill_value=0)
    df[standard_cols] = standard_scaler.transform(df[standard_cols])
    df[robust_cols] = robust_scaler.transform(df[robust_cols])
    return df

def predict(data):
    input_data = preprocess_input(data)
    pred = model.predict(input_data)[0]
    return "Canceled" if pred == 1 else "Not Canceled"

st.title("Hotel Booking Cancellation Prediction")

with st.form("prediction_form"):
    no_of_adults = st.number_input("Number of Adults", value=2)
    no_of_children = st.number_input("Number of Children", value=0)
    no_of_weekend_nights = st.number_input("Weekend Nights", value=1)
    no_of_week_nights = st.number_input("Week Nights", value=2)
    type_of_meal_plan = st.selectbox("Meal Plan", ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'])
    required_car_parking_space = st.selectbox("Parking Space Required", [0, 1])
    room_type_reserved = st.selectbox("Room Type", ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7'])
    lead_time = st.number_input("Lead Time", value=30)

    # ✅ Dynamic year selection
    current_year = datetime.datetime.now().year
    arrival_year = st.selectbox("Arrival Year", list(range(2017, current_year + 1)))

    arrival_month = st.selectbox("Arrival Month", list(range(1, 13)))
    arrival_date = st.selectbox("Arrival Date", list(range(1, 32)))
    market_segment_type = st.selectbox("Market Segment", ['Online', 'Offline', 'Corporate', 'Complementary', 'Aviation'])
    repeated_guest = st.selectbox("Repeated Guest", [0, 1])
    no_of_previous_cancellations = st.number_input("Previous Cancellations", value=0)
    no_of_previous_bookings_not_canceled = st.number_input("Previous Bookings Not Canceled", value=0)
    avg_price_per_room = st.number_input("Average Price Per Room", value=100.0)
    no_of_special_requests = st.number_input("Special Requests", value=0)
    
    submitted = st.form_submit_button("Predict")
    if submitted:
        data = {
            'no_of_adults': no_of_adults,
            'no_of_children': no_of_children,
            'no_of_weekend_nights': no_of_weekend_nights,
            'no_of_week_nights': no_of_week_nights,
            'type_of_meal_plan': type_of_meal_plan,
            'required_car_parking_space': required_car_parking_space,
            'room_type_reserved': room_type_reserved,
            'lead_time': lead_time,
            'arrival_year': arrival_year,
            'arrival_month': arrival_month,
            'arrival_date': arrival_date,
            'market_segment_type': market_segment_type,
            'repeated_guest': repeated_guest,
            'no_of_previous_cancellations': no_of_previous_cancellations,
            'no_of_previous_bookings_not_canceled': no_of_previous_bookings_not_canceled,
            'avg_price_per_room': avg_price_per_room,
            'no_of_special_requests': no_of_special_requests
        }
        st.write("Input Data:")
        st.dataframe(pd.DataFrame([data]))
        result = predict(data)
        st.subheader(f"Prediction Result: {result}")

# Test Case - Not Canceled
if st.button("Test Case - Not Canceled"):
    test_data_1 = {
        'no_of_adults': 2,
        'no_of_children': 0,
        'no_of_weekend_nights': 1,
        'no_of_week_nights': 2,
        'type_of_meal_plan': 'Meal Plan 1',
        'required_car_parking_space': 1,
        'room_type_reserved': 'Room_Type 1',
        'lead_time': 20,
        'arrival_year': current_year,
        'arrival_month': 8,
        'arrival_date': 15,
        'market_segment_type': 'Online',
        'repeated_guest': 0,
        'no_of_previous_cancellations': 0,
        'no_of_previous_bookings_not_canceled': 1,
        'avg_price_per_room': 100.0,
        'no_of_special_requests': 1
    }
    st.write("Test Case - Not Canceled Input:")
    st.dataframe(pd.DataFrame([test_data_1]))
    result = predict(test_data_1)
    st.subheader(f"Test Case Result: {result}")

# Test Case - Canceled
if st.button("Test Case - Canceled"):
    test_data_2 = {
        'no_of_adults': 2,
        'no_of_children': 2,
        'no_of_weekend_nights': 3,
        'no_of_week_nights': 5,
        'type_of_meal_plan': 'Not Selected',
        'required_car_parking_space': 0,
        'room_type_reserved': 'Room_Type 6',
        'lead_time': 200,
        'arrival_year': current_year,
        'arrival_month': 12,
        'arrival_date': 28,
        'market_segment_type': 'Offline',
        'repeated_guest': 0,
        'no_of_previous_cancellations': 2,
        'no_of_previous_bookings_not_canceled': 0,
        'avg_price_per_room': 130.0,
        'no_of_special_requests': 0
    }
    st.write("Test Case - Canceled Input:")
    st.dataframe(pd.DataFrame([test_data_2]))
    result = predict(test_data_2)
    st.subheader(f"Test Case Result: {result}")
