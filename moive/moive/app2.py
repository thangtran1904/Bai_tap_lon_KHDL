import streamlit as st
import pandas as pd
import requests
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from difflib import get_close_matches
import os

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# -----------------------------------------------
# 📋 Session State Setup
# -----------------------------------------------
if "shown_count" not in st.session_state:
    st.session_state.shown_count = 5
if "recommend_trigger" not in st.session_state:
    st.session_state.recommend_trigger = False
if "last_movie" not in st.session_state:
    st.session_state.last_movie = None

# -----------------------------------------------
# 🎞️ Lấy poster phim từ TMDb
# -----------------------------------------------
@st.cache_data
def fetch_poster(title):
    clean_title = re.sub(r"\s+\(\d{4}\)", "", title)
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={clean_title}"
    response = requests.get(url)
    data = response.json()
    if data.get("results"):
        poster_path = data["results"][0].get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return None

# -----------------------------------------------
# 🎬 Thiết lập trang
# -----------------------------------------------
st.set_page_config(page_title="Gợi ý phim", layout="centered")
st.title("🎬 Hệ thống gợi ý phim (Dựa trên nội dung)")
st.write("Nhập tên phim bạn thích, hệ thống sẽ gợi ý các phim tương tự dựa trên thể loại.")

# -----------------------------------------------
# 📊 Tải dữ liệu phim
# -----------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/movies.csv")

movies_df = load_data()
movie_titles = movies_df['title'].tolist()

# -----------------------------------------------
# 🧠 Tạo ma trận tương đồng theo thể loại
# -----------------------------------------------
@st.cache_resource
def build_model(data):
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split('|'), stop_words='english')
    genre_matrix = vectorizer.fit_transform(data['genres'])
    similarity = cosine_similarity(genre_matrix)
    return similarity

similarity_matrix = build_model(movies_df)

# -----------------------------------------------
# 🔍 Nhập tên phim cần tìm
# -----------------------------------------------
search_query = st.text_input("🔎 Nhập tên phim:")

matched_movie = None
if search_query:
    title_map = {title.lower(): title for title in movie_titles}
    lowered_titles = list(title_map.keys())
    query = search_query.lower()

    matches_lower = get_close_matches(query, lowered_titles, n=5, cutoff=0.4)
    partial_matches = [title for title in lowered_titles if query in title]
    combined_matches = list(dict.fromkeys(matches_lower + partial_matches))
    matches = [title_map[match] for match in combined_matches]

    if matches:
        new_match = st.selectbox("🎞 Chọn tên phim gần đúng:", matches)

        if new_match != st.session_state.last_movie:
            st.session_state.recommend_trigger = False
            st.session_state.shown_count = 5

        matched_movie = new_match
        st.session_state.last_movie = matched_movie
    else:
        st.warning("Không tìm thấy phim phù hợp. Hãy thử tên khác.")

# -----------------------------------------------
# 🎯 Hiển thị gợi ý
# -----------------------------------------------
if matched_movie:
    if st.button("🔍 Xem gợi ý phim", key="show_initial"):
        st.session_state.shown_count = 5
        st.session_state.recommend_trigger = True

    if st.session_state.recommend_trigger:
        idx = movies_df[movies_df['title'] == matched_movie].index[0]
        sim_scores = list(enumerate(similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        searched_poster = fetch_poster(matched_movie)
        st.subheader("🎯 Phim bạn đã chọn:")
        col1, col2 = st.columns([1, 4])
        with col1:
            if searched_poster:
                st.image(searched_poster, width=100)
            else:
                st.text("🎞️ Không có poster")
        with col2:
            st.markdown(f"**{matched_movie}**")
            st.caption("Điểm tương đồng: `1.00`")

        recommendations = [
            (i, score) for i, score in sim_scores if movies_df.iloc[i]["title"] != matched_movie
        ]

        st.markdown("---")
        st.subheader("📽️ Bạn cũng có thể thích những phim sau:")

        max_to_show = 50
        shown = 0
        for movie_idx, score in recommendations:
            if shown >= st.session_state.shown_count or shown >= max_to_show:
                break

            title = movies_df.iloc[movie_idx]["title"]
            poster_url = fetch_poster(title)

            col1, col2 = st.columns([1, 4])
            with col1:
                if poster_url:
                    st.image(poster_url, width=100)
                else:
                    st.text("🎞️ Không có poster")
            with col2:
                st.markdown(f"**{title}**")
                st.caption(f"Điểm tương đồng: `{score:.2f}`")

            shown += 1

        if shown < len(recommendations) and shown < max_to_show:
            if st.button("🔄 Xem thêm gợi ý", key="show_more"):
                st.session_state.shown_count += 5
                st.rerun()
