import pandas as pd
import pickle
from surprise import Dataset, Reader, SVD

# Load dữ liệu
movies_df = pd.read_csv("data/movies.csv")
ratings_df = pd.read_csv("data/ratings.csv")
links_df = pd.read_csv("data/links.csv")
movies_with_links_df = movies_df.merge(links_df, on="movieId", how="left")

# Train mô hình SVD
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()
model = SVD()
model.fit(trainset)

# Lưu mô hình vào file
with open("trained_svd_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Đã train và lưu mô hình SVD vào trained_svd_model.pkl")
