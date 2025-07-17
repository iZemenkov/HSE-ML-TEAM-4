import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Загрузка модели
model = joblib.load('model_pipeline.pkl')

st.title("🎵 Определение музыкального жанра")

option = st.radio("Выберите способ ввода данных:", ["📝 Ввод вручную", "📂 Загрузка CSV"])

# === Общие словари для обоих путей ===
key_order = {'C':0,'C#':1,'D':2,'D#':3,'E':4,'F':5,'F#':6,'G':7,'G#':8,'A':9,'A#':10,'B':11}
mode_map  = {'Major':1, 'Minor':0}

# === Функция обработки входных данных ===
def preprocess(df):
    df = df.copy()

    # Базовая обработка
    df['duration_sec'] = df['duration_ms'] / 1000
    df['log_duration'] = np.log1p(df['duration_sec'])
    df['sqrt_duration'] = np.sqrt(np.maximum(df['duration_sec'], 0))
    df['energy_dance'] = df['energy'] * df['danceability']
    df['valence_energy'] = df['valence'] * df['energy']
    df['liveness_speechiness'] = df['liveness'] * df['speechiness']
    df['speech_instr_sum'] = df['speechiness'] + df['instrumentalness']
    df['loudness_energy_ratio'] = df['loudness'] / (df['energy'] + 1e-6)
    df['acoustic_energy_ratio'] = df['acousticness'] / (df['energy'] + 1e-6)
    df['loudness_duration'] = df['loudness'] * df['duration_sec']
    df['tempo_low'] = (df['tempo'] < 90).astype(int)
    df['tempo_mid'] = ((df['tempo'] >= 90) & (df['tempo'] < 150)).astype(int)
    df['tempo_high'] = (df['tempo'] >= 150).astype(int)

    df['key'] = df['key'].astype(str)
    df['mode'] = df['mode'].astype(str)

    key_order = {'C':0,'C#':1,'D':2,'D#':3,'E':4,'F':5,'F#':6,'G':7,'G#':8,'A':9,'A#':10,'B':11}
    mode_map  = {'Major':1, 'Minor':0}
    df['key_num'] = df['key'].map(key_order)
    df['mode_num'] = df['mode'].map(mode_map)

    df['title_len'] = df['track_name'].str.len()
    df['title_word_cnt'] = df['track_name'].str.split().apply(len)
    df['has_remix'] = df['track_name'].str.contains('remix', case=False).astype(int)
    df['has_live'] = df['track_name'].str.contains('live', case=False).astype(int)

    # Удаляем ненужные признаки
    to_drop = ['tempo_low', 'tempo_mid', 'duration_sec', 'sqrt_duration',
               'acoustic_energy_ratio', 'loudness_energy_ratio']
    df.drop(columns=to_drop, inplace=True, errors='ignore')

    # === Обработка пропущенных значений ===
    df.fillna(0, inplace=True)

    return df

# === Ввод вручную ===
if option == "📝 Ввод вручную":
    st.subheader("Введите характеристики трека:")

    instance_id = st.text_input("Instance ID", "0001")
    track_name = st.text_input("Track Name", "My Awesome Track")
    obtained_date = st.date_input("Obtained Date")
    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
    danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
    duration_ms = st.number_input("Duration (ms)", min_value=10000, max_value=600000, value=180000)
    energy = st.slider("Energy", 0.0, 1.0, 0.5)
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.5)
    key = st.selectbox("Key", list(key_order.keys()))
    liveness = st.slider("Liveness", 0.0, 1.0, 0.2)
    loudness = st.number_input("Loudness (dB)", -60.0, 5.0, -10.0)
    mode_val = st.selectbox("Mode", list(mode_map.keys()))
    speechiness = st.slider("Speechiness", 0.0, 1.0, 0.5)
    tempo = st.number_input("Tempo (BPM)", 40.0, 250.0, 120.0)
    valence = st.slider("Valence", 0.0, 1.0, 0.5)

    input_df = pd.DataFrame([{
        'instance_id': instance_id,
        'track_name': track_name,
        'obtained_date': obtained_date,
        'acousticness': acousticness,
        'danceability': danceability,
        'duration_ms': duration_ms,
        'energy': energy,
        'instrumentalness': instrumentalness,
        'key': key,
        'liveness': liveness,
        'loudness': loudness,
        'mode': mode_val,
        'speechiness': speechiness,
        'tempo': tempo,
        'valence': valence
    }])

    input_df = input_df.drop(columns=["instance_id", "obtained_date"])
    processed_df = preprocess(input_df)

    if st.button("Предсказать жанр"):
        prediction = model.predict(processed_df)[0]
        st.success(f"🎧 Предсказанный жанр: **{prediction}**")

# === Загрузка CSV ===
else:
    st.subheader("Загрузите CSV-файл с треками:")
    uploaded_file = st.file_uploader("Выберите CSV-файл", type=["csv"])

    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)

            if not {'track_name', 'acousticness', 'danceability', 'duration_ms', 'energy',
                    'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
                    'speechiness', 'tempo', 'valence'}.issubset(data.columns):
                st.error("❌ Файл не содержит нужных столбцов.")
            else:
                processed = preprocess(data)
                predictions = model.predict(processed)
                data['predicted_genre'] = predictions
                st.success("🎯 Предсказания выполнены!")
                st.dataframe(data[['track_name', 'predicted_genre']])

                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button("💾 Скачать с результатами", csv, "predicted_tracks.csv", "text/csv")

        except Exception as e:
            st.error(f"Произошла ошибка при обработке файла: {e}")
