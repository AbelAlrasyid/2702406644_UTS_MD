import streamlit as st
import numpy as np
import pandas as pd
import pickle
import gzip

# --- Load model dan scaler ---
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

# --- Form Input User ---
st.title("📊 Prediksi Pembatalan Pemesanan Hotel")

with st.form("user_form"):
    st.subheader("📝 Isi data pemesanan:")

    no_of_adults = st.number_input("Jumlah Dewasa", min_value=0, value=2)
    no_of_children = st.number_input("Jumlah Anak", min_value=0, value=0)
    no_of_weekend_nights = st.number_input("Malam Akhir Pekan", min_value=0, value=1)
    no_of_week_nights = st.number_input("Malam Hari Kerja", min_value=0, value=2)
    required_car_parking_space = st.selectbox("Perlu Parkir?", [0, 1])
    lead_time = st.number_input("Lead Time (hari)", min_value=0, value=30)
    arrival_year = st.selectbox("Tahun Kedatangan", [2017, 2018])
    arrival_month = st.number_input("Bulan Kedatangan", 1, 12, value=5)
    arrival_date = st.number_input("Tanggal Kedatangan", 1, 31, value=15)
    repeated_guest = st.selectbox("Tamu Berulang?", [0, 1])
    no_of_previous_cancellations = st.number_input("Pembatalan Sebelumnya", 0, value=0)
    no_of_previous_bookings_not_canceled = st.number_input("Booking Sebelumnya Tidak Dibatalkan", 0, value=0)
    avg_price_per_room = st.number_input("Rata-rata Harga per Kamar", value=100.0)
    no_of_special_requests = st.number_input("Jumlah Permintaan Khusus", 0, value=0)

    type_of_meal_plan = st.selectbox("Paket Makan", ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'])
    room_type_reserved = st.selectbox("Tipe Kamar", ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7'])
    market_segment_type = st.selectbox("Segment Pasar", ['Online', 'Offline', 'Corporate', 'Complementary', 'Aviation'])

    submit = st.form_submit_button("🔮 Prediksi")

if submit:
    # --- Buat DataFrame dari input ---
    input_data = pd.DataFrame([{
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
    }])

    # --- One-hot encoding (dummies) ---
    input_encoded = pd.get_dummies(input_data, columns=[
        'type_of_meal_plan', 'room_type_reserved', 'market_segment_type'
    ])
    input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

    # --- Scaling ---
    input_encoded[['arrival_month', 'arrival_date']] = standard_scaler.transform(
        input_encoded[['arrival_month', 'arrival_date']]
    )

    input_encoded[[
        'no_of_adults', 'no_of_children', 'no_of_weekend_nights',
        'no_of_week_nights', 'required_car_parking_space', 'lead_time',
        'arrival_year', 'repeated_guest', 'no_of_previous_cancellations',
        'no_of_previous_bookings_not_canceled', 'avg_price_per_room',
        'no_of_special_requests'
    ]] = robust_scaler.transform(
        input_encoded[[
            'no_of_adults', 'no_of_children', 'no_of_weekend_nights',
            'no_of_week_nights', 'required_car_parking_space', 'lead_time',
            'arrival_year', 'repeated_guest', 'no_of_previous_cancellations',
            'no_of_previous_bookings_not_canceled', 'avg_price_per_room',
            'no_of_special_requests'
        ]]
    )

    # --- Prediksi ---
    prediction = model.predict(input_encoded)[0]
    proba = model.predict_proba(input_encoded)[0][prediction]

    label = "❌ Dibatalkan" if prediction == 1 else "✅ Tidak Dibatalkan"
    st.success(f"📢 Prediksi: **{label}** (Probabilitas: {proba:.2f})")
