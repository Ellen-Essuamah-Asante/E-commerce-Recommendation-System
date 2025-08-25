from scipy.sparse import csr_matrix
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

st.title("🛒 Hybrid Recommendation System")
st.write("Get personalized product recommendations using Collaborative + Content-Based filtering.")

# =======================
# Load Data (from sampled CSVs)
# =======================


@st.cache_data
def load_data():
    events = pd.read_csv("data/events_sample.csv")
    item_props = pd.read_csv("data/item_properties_sample.csv")
    category_tree = pd.read_csv("data/category_tree_sample.csv")
    return events, item_props, category_tree


events, item_props, category_tree = load_data()
st.success("✅ Sampled data loaded successfully.")
st.write("Events preview:", events.head())

# =======================
# Preprocess Data
# =======================
weights = {"view": 1, "addtocart": 3, "transaction": 5}
events["event_strength"] = events["event"].map(weights)
events = events.dropna(subset=["visitorid", "itemid"])

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
            tfidf_matrix[tfidf_idx], n_neighbors=6, return_distance=True
        )
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

    interacted_items = user_item_strength[user_item_strength["visitorid"] == user_id]["itemid"].map(
        item_mapping)
    cb_scores = get_cb_scores(interacted_items, len(item_ids))

    final_scores = alpha * cf_scores + (1 - alpha) * cb_scores
    top_items_idx = np.argsort(-final_scores)[:top_n]
    return [inv_item_mapping[idx] for idx in top_items_idx]


# =======================
# Streamlit UI
# =======================
user_input = st.text_input("Enter User ID:", "")
if user_input:
    try:
        user_id = int(user_input)
        recs = hybrid_recommend(user_id, top_n=10, alpha=0.6)
        st.write(f"### 🎯 Top Recommendations for User {user_id}")
        for r in recs:
            st.write(f"- Item {r}")
    except ValueError:
        st.error("⚠️ Please enter a valid numeric User ID")
