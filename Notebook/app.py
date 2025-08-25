from scipy.sparse import csr_matrix
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# =======================
# Load / Prepare Data
# =======================
# (Replace these with your real preprocessed data)
events = pd.read_csv(
    r"C:\Users\mbs-p\Desktop\E-commerce-Recommendation-System\Notebook\events_clean.csv")
item_props = pd.read_csv(
    r"C:\Users\mbs-p\Desktop\E-commerce-Recommendation-System\Notebook\item_properties_clean.csv")
category_tree = pd.read_csv(
    r"C:\Users\mbs-p\Desktop\E-commerce-Recommendation-System\Notebook\category_tree.csv")

# Assign weights
weights = {"view": 1, "addtocart": 3, "transaction": 5}
events["event_strength"] = events["event"].map(weights)
events = events.dropna(subset=["visitorid", "itemid"])

# User-item matrix
user_item_strength = (
    events.groupby(["visitorid", "itemid"])["event_strength"]
    .sum()
    .reset_index()
)

# Create mapping
user_ids = user_item_strength["visitorid"].unique()
item_ids = user_item_strength["itemid"].unique()
user_mapping = {u: idx for idx, u in enumerate(user_ids)}
item_mapping = {i: idx for idx, i in enumerate(item_ids)}
inv_item_mapping = {v: k for k, v in item_mapping.items()}

# Build interaction matrix
rows = user_item_strength["visitorid"].map(user_mapping)
cols = user_item_strength["itemid"].map(item_mapping)
values = user_item_strength["event_strength"]
X = csr_matrix((values, (rows, cols)), shape=(len(user_ids), len(item_ids)))

# =======================
# Collaborative Filtering (SVD)
# =======================
svd = TruncatedSVD(n_components=50, random_state=42)
user_factors = svd.fit_transform(X)
item_factors = svd.components_.T

# =======================
# Content-Based (TF-IDF)
# =======================
item_features = item_props.groupby("itemid")["num_value"] \
    .apply(lambda x: " ".join(x.dropna().astype(str))) \
    .reset_index()

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(item_features["num_value"].astype(str))

tfidf_item_ids = item_features["itemid"].tolist()
tfidf_item_mapping = {item_id: idx for idx,
                      item_id in enumerate(tfidf_item_ids)}

knn = NearestNeighbors(metric="cosine", algorithm="brute",
                       n_neighbors=20, n_jobs=-1)
knn.fit(tfidf_matrix)

# =======================
# Hybrid Recommendation Function
# =======================


def get_cb_scores(interacted_items_idx, num_items):
    cb_scores = np.zeros(num_items)
    for idx in interacted_items_idx:
        item_id = inv_item_mapping[idx]
        if item_id not in tfidf_item_mapping:
            continue
        tfidf_idx = tfidf_item_mapping[item_id]

        sims, indices = knn.kneighbors(
            tfidf_matrix[tfidf_idx], n_neighbors=6, return_distance=True)
        sims = 1 - sims.flatten()

        for neigh_idx, sim in zip(indices.flatten(), sims):
            neigh_item_id = tfidf_item_ids[neigh_idx]
            if neigh_item_id in item_mapping:
                cf_idx = item_mapping[neigh_item_id]
                cb_scores[cf_idx] += sim
    return cb_scores


def hybrid_recommend(user_id, top_n=10, alpha=0.5):
    if user_id not in user_mapping:
        return ["Popular_" + str(i) for i in range(top_n)]

    user_idx = user_mapping[user_id]
    cf_scores = np.dot(user_factors[user_idx], item_factors.T)

    user_row = X.getrow(user_idx).toarray().ravel()
    interacted_items_idx = np.where(user_row > 0)[0]
    cb_scores = get_cb_scores(interacted_items_idx, item_factors.shape[0]) if len(
        interacted_items_idx) > 0 else np.zeros(item_factors.shape[0])

    hybrid_scores = alpha * cf_scores + (1 - alpha) * cb_scores
    hybrid_scores[interacted_items_idx] = -np.inf

    recommended_idx = np.argsort(-hybrid_scores)[:top_n]
    return [inv_item_mapping[i] for i in recommended_idx]


# =======================
# Streamlit UI
# =======================
st.title("🛒 Hybrid Recommendation System")
st.write("Get personalized product recommendations using Collaborative + Content-Based filtering.")

# Select user
user_id = st.selectbox("Select User ID", user_ids)

# Hyperparameter
alpha = st.slider("Weight for Collaborative Filtering (α)", 0.0, 1.0, 0.5, 0.1)
top_n = st.number_input("Number of Recommendations",
                        min_value=1, max_value=20, value=5)

if st.button("Get Recommendations"):
    recs = hybrid_recommend(user_id, top_n=top_n, alpha=alpha)
    st.subheader(f"Recommendations for User {user_id}")
    for i, r in enumerate(recs, 1):
        st.write(f"{i}. Item {r}")
