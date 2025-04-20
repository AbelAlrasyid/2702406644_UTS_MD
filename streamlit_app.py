import streamlit as st
import pickle
import numpy as np
import pandas as pd

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
        columns = pickle.load(f)
    return model, standard_scaler, robust_scaler, columns

model, standard_scaler, robust_scaler, all_columns = load_all()

# Kategori (disesuaikan dari training set)
meal_options = ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected']
room_options = ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7']
market_options = ['Offline', 'Online', 'Corporate', 'Complementary', 'Aviation']

# Input manual
st.title("🚪 Prediksi Pembatalan Pemesanan Hotel")

with st.form("input_form"):
    st.subheader("📝 Masukkan Informasi Booking:")
    
    no_of_adults = st.number_input("Jumlah Dewasa", min_value=0, value=2)
    no_of_children = st.number_input("Jumlah Anak", min_value=0, value=0)
    no_of_weekend_nights = st.number_input("Jumlah Malam Akhir Pekan", min_value=0, value=1)
    no_of_week_nights = st.number_input("Jumlah Malam Hari Kerja", min_value=0, value=2)
    required_car_parking_space = st.selectbox("Butuh Parkir?", [0, 1])
    lead_time = st.number_input("Lead Time (hari sebelum menginap)", min_value=0, value=30)
    arrival_year = st.selectbox("Tahun Kedatangan", [2017])
    arrival_month = st.number_input("Bulan Kedatangan", min_value=1, max_value=12, value=7)
    arrival_date = st.number_input("Tanggal Kedatangan", min_value=1, max_value=31, value=15)
    repeated_guest = st.selectbox("Tamu Langganan?", [0, 1])
    no_of_previous_cancellations = st.number_input("Pembatalan Sebelumnya", min_value=0, value=0)
    no_of_previous_bookings_not_canceled = st.number_input("Booking Sebelumnya Tidak Dibatalkan", min_value=0, value=0)
    avg_price_per_room = st.number_input("Harga Rata-rata per Malam", min_value=0.0, value=100.0)
    no_of_special_requests = st.number_input("Jumlah Permintaan Khusus", min_value=0, value=0)
    
    type_of_meal_plan = st.selectbox("Paket Makan", meal_options)
    room_type_reserved = st.selectbox("Tipe Kamar", room_options)
    market_segment_type = st.selectbox("Tipe Segmen Market", market_options)

    submitted = st.form_submit_button("Prediksi")

# Mapping input ke dataframe
def process_input(data_dict):
    df = pd.DataFrame([data_dict])
    cat_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
    standard_cols = ['arrival_month', 'arrival_date']
    robust_cols = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights',
                   'no_of_week_nights', 'required_car_parking_space', 'lead_time',
                   'arrival_year', 'repeated_guest', 'no_of_previous_cancellations',
                   'no_of_previous_bookings_not_canceled', 'avg_price_per_room',
                   'no_of_special_requests']

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    df = df.reindex(columns=all_columns, fill_value=0)

    df[standard_cols] = standard_scaler.transform(df[standard_cols])
    df[robust_cols] = robust_scaler.transform(df[robust_cols])

    return df

# Test case 1 dan 2
test_cases = [
    {
        'no_of_adults': 2,
        'no_of_children': 0,
        'no_of_weekend_nights': 1,
        'no_of_week_nights': 2,
        'required_car_parking_space': 0,
        'lead_time': 30,
        'arrival_year': 2017,
        'arrival_month': 7,
        'arrival_date': 15,
        'repeated_guest': 0,
        'no_of_previous_cancellations': 0,
        'no_of_previous_bookings_not_canceled': 0,
        'avg_price_per_room': 100.0,
        'no_of_special_requests': 1,
        'type_of_meal_plan': 'Meal Plan 1',
        'room_type_reserved': 'Room_Type 1',
        'market_segment_type': 'Online'
    },
    {
        test_case_2 = {
        'no_of_adults': 2,
        'no_of_children': 2,
        'no_of_weekend_nights': 3,
        'no_of_week_nights': 5,
        'type_of_meal_plan': 'Not Selected',
        'required_car_parking_space': 0,
        'room_type_reserved': 'Room_Type 6',
        'lead_time': 200,
        'arrival_year': 2018,
        'arrival_month': 12,
        'arrival_date': 28,
        'market_segment_type': 'Offline',
        'repeated_guest': 0,
        'no_of_previous_cancellations': 2,
        'no_of_previous_bookings_not_canceled': 0,
        'avg_price_per_room': 130.0,
        'no_of_special_requests': 0
}

    }
]

if submitted:
    user_input = {
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

    input_df = process_input(user_input)
    pred = model.predict(input_df)[0]
    label = "❌ Dibatalkan" if pred == 1 else "✅ Tidak Dibatalkan"
    st.success(f"Hasil Prediksi: {label}")

# Test case section
st.markdown("---")
st.subheader("🧪 Test Case")

for i, test_case in enumerate(test_cases):
    test_input = process_input(test_case)
    pred = model.predict(test_input)[0]
    label = "❌ Dibatalkan" if pred == 1 else "✅ Tidak Dibatalkan"
    with st.expander(f"Test Case {i+1}"):
        st.write(pd.DataFrame([test_case]))
        st.success(f"Hasil Prediksi: {label}")
