import streamlit as st
import pandas as pd
import pickle
import requests
import numpy as np

st.set_page_config(page_title="Gợi Ý Phim Với Đánh Giá Người Dùng Mới", layout="centered")
st.title("🎬 Hệ Thống Gợi Ý Phim - Đánh Giá Cá Nhân")

# Load dữ liệu và model
@st.cache_data
def load_data():
    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")
    links = pd.read_csv("data/links.csv")
    with open("trained_svd_model.pkl", "rb") as f:
        model = pickle.load(f)
    return movies, ratings, links, model

movies_df, ratings_df, links_df, svd_model = load_data()

# Lấy user_id max hiện tại để tạo user giả
max_user_id = ratings_df['userId'].max()
new_user_id = max_user_id + 1

# Chọn 20 phim phổ biến nhất để user đánh giá (có thể thay đổi)
popular_movies = ratings_df['movieId'].value_counts().head(20).index.tolist()
popular_movies_df = movies_df[movies_df['movieId'].isin(popular_movies)]

st.write("Vui lòng đánh giá 5 phim dưới đây (1 - 5 sao):")

# Khởi tạo session state cho lưu điểm đánh giá nếu chưa có
if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = {}

# Hiển thị các phim để user đánh giá
for idx, row in popular_movies_df.iterrows():
    movie_id = row['movieId']
    title = row['title']
    rating = st.slider(label=title, min_value=0, max_value=5, step=1, key=f"rate_{movie_id}")
    if rating > 0:
        st.session_state.user_ratings[movie_id] = rating
    elif movie_id in st.session_state.user_ratings:
        del st.session_state.user_ratings[movie_id]

# Kiểm tra đã đánh giá đủ 5 phim chưa
if len(st.session_state.user_ratings) < 5:
    st.warning(f"Vui lòng đánh giá ít nhất 5 phim. Hiện tại bạn đã đánh giá {len(st.session_state.user_ratings)} phim.")
else:
    if st.button("Nhận gợi ý phim cho bạn"):
        user_ratings = st.session_state.user_ratings

        # Tạo dataframe ratings mới cho user giả
        new_ratings = pd.DataFrame({
            'userId': new_user_id,
            'movieId': list(user_ratings.keys()),
            'rating': list(user_ratings.values())
        })

        # Tạo danh sách phim để dự đoán điểm (những phim user chưa đánh giá)
        all_movie_ids = ratings_df['movieId'].unique()
        unrated_movie_ids = [mid for mid in all_movie_ids if mid not in user_ratings]

        # Dự đoán điểm cho phim chưa đánh giá
        predictions = []
        for movie_id in unrated_movie_ids:
            pred = svd_model.predict(new_user_id, movie_id)
            predictions.append((movie_id, pred.est))

        # Lấy top 10 phim có điểm dự đoán cao nhất
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_preds = predictions[:10]

        st.subheader("🎯 Phim được gợi ý cho bạn:")

        def fetch_poster(tmdb_id):
            if pd.isna(tmdb_id):
                return "https://via.placeholder.com/500x750?text=No+Image"
            url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key=YOUR_TMDB_API_KEY"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                poster_path = data.get("poster_path")
                if poster_path:
                    return f"https://image.tmdb.org/t/p/w500{poster_path}"
            return "https://via.placeholder.com/500x750?text=No+Image"

        # Hiển thị kết quả
        for movie_id, score in top_preds:
            movie_info = movies_df[movies_df['movieId'] == movie_id].iloc[0]
            title = movie_info['title']
            tmdb_id = links_df[links_df['movieId'] == movie_id]['tmdbId'].values[0]
            poster_url = fetch_poster(tmdb_id)

            col1, col2 = st.columns([1, 4])
            with col1:
                st.image(poster_url, width=100)
            with col2:
                st.markdown(f"**{title}**")
                st.caption(f"Điểm dự đoán: {score:.2f}")
