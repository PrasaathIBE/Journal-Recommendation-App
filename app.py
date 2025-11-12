# import streamlit as st
# import pandas as pd
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

# # ============================================================
# # 🔹 1. App Configuration
# # ============================================================
# st.set_page_config(page_title="Semantic Journal Recommendation App", layout="centered")
# st.title("📘 Semantic Journal Recommendation System")
# st.markdown("### Suggests the most suitable journals based on your paper title")

# # ============================================================
# # 🔹 2. Cache Model and Data Loading
# # ============================================================
# @st.cache_resource
# def load_model():
#     model = SentenceTransformer("all-MiniLM-L6-v2")
#     return model

# @st.cache_data
# def load_journal_data(uploaded_file):
#     df = pd.read_excel(uploaded_file)
#     df["Journal_Name"] = df["Journal_Name"].fillna("").astype(str)
#     df["Special_Issue_Name"] = df["Special_Issue_Name"].fillna("").astype(str)
#     df["Special_Issue_keywords"] = df["Special_Issue_keywords"].fillna("").astype(str)
#     # Weighted combination: keywords prioritized
#     df["combined_text"] = (
#         df["Special_Issue_keywords"] + " " +
#         df["Special_Issue_keywords"] + " " +
#         df["Special_Issue_Name"]
#     )
#     return df

# # ============================================================
# # 🔹 3. Sidebar Inputs
# # ============================================================
# st.sidebar.header("🧾 Input Options")
# uploaded_file = st.sidebar.file_uploader("Upload Journal Sheet (.xlsx)", type=["xlsx"])
# paper_title = st.sidebar.text_area("Enter Paper Title", placeholder="Paste your paper title here...")

# # ============================================================
# # 🔹 4. Load Model (Cached)
# # ============================================================
# with st.spinner("Loading semantic model... Please wait..."):
#     model = load_model()
# st.success("Model loaded and ready!")

# # ============================================================
# # 🔹 5. Process When Inputs Are Provided
# # ============================================================
# if uploaded_file is not None and paper_title.strip() != "":
#     journals_df = load_journal_data(uploaded_file)

#     st.info(f"✅ Loaded {len(journals_df)} journal entries for matching.")
#     threshold = 0.5
#     top_n = 6

#     with st.spinner("Analyzing and finding suitable journals..."):
#         # Encode the single paper title
#         paper_embedding = model.encode([paper_title])
#         # Encode journal combined text
#         journal_embeddings = model.encode(journals_df["combined_text"].tolist(), show_progress_bar=False)

#         # Compute cosine similarity
#         similarity_scores = cosine_similarity(paper_embedding, journal_embeddings)[0]
#         score_list = list(enumerate(similarity_scores))
#         sorted_scores = sorted(score_list, key=lambda x: x[1], reverse=True)

#         # Filter based on threshold
#         filtered_scores = [(idx, score) for idx, score in sorted_scores if score >= threshold]
#         if len(filtered_scores) < top_n:
#             filtered_scores = sorted_scores[:top_n]

#         top_indices = [idx for idx, _ in filtered_scores]
#         top_journals = journals_df.iloc[top_indices]["Journal_Name"].tolist()
#         top_scores = [round(score, 3) for _, score in filtered_scores]

#         # Deduplicate journals while preserving order
#         unique_recommendations = []
#         for j, s in zip(top_journals, top_scores):
#             if j not in [r["Journal_Name"] for r in unique_recommendations]:
#                 unique_recommendations.append({"Journal_Name": j, "Similarity": s})
#         unique_recommendations = unique_recommendations[:top_n]

#     # ============================================================
#     # 🔹 6. Display Results
#     # ============================================================
#     st.subheader("🎯 Top Recommended Journals:")
#     if unique_recommendations:
#         results_df = pd.DataFrame(unique_recommendations)
#         st.table(results_df)
#     else:
#         st.warning("No strong matches found (try lowering similarity threshold or using broader keywords).")

# else:
#     st.info("⬅️ Please upload a journal sheet and enter a paper title to begin.")

# # ============================================================
# # 🔹 7. Footer
# # ============================================================
# st.markdown("---")
# st.caption("Developed by Prasaath Ravichandran • Uses Sentence-BERT for semantic similarity matching")


import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# 1️⃣ Streamlit Page Setup
# ============================================================
st.set_page_config(page_title="Semantic Journal Recommendation App", layout="centered")
st.title("📘 Semantic Journal Recommendation System (Enhanced Accuracy)")
st.markdown("### Uses a research-paper-tuned model (SPECTER) for precise semantic matching")

# ============================================================
# 2️⃣ Cache Model & Data Loading
# ============================================================
@st.cache_resource
def load_model():
    model = SentenceTransformer("allenai-specter")  # Scientific text-tuned model
    return model

@st.cache_data
def load_journal_data(uploaded_file):
    df = pd.read_excel(uploaded_file)
    df["Journal_Name"] = df["Journal_Name"].fillna("").astype(str)
    df["Special_Issue_Name"] = df["Special_Issue_Name"].fillna("").astype(str)
    df["Special_Issue_keywords"] = df["Special_Issue_keywords"].fillna("").astype(str)

    # Create weighted, context-enriched journal text
    df["combined_text"] = (
        "This journal special issue discusses topics such as "
        + df["Special_Issue_keywords"] + " " + df["Special_Issue_keywords"]
        + " and focuses on areas including " + df["Special_Issue_Name"]
        + ". It covers advanced methods in AI, IoT, automation, and intelligent systems."
    )
    return df

# ============================================================
# 3️⃣ Sidebar Inputs
# ============================================================
st.sidebar.header("🧾 Input Options")
uploaded_file = st.sidebar.file_uploader("Upload Journal Sheet (.xlsx)", type=["xlsx"])
paper_title = st.sidebar.text_area("Enter Paper Title", placeholder="Paste your paper title here...")

# ============================================================
# 4️⃣ Load Model
# ============================================================
with st.spinner("Loading semantic model (SPECTER)... Please wait..."):
    model = load_model()
st.success("✅ Model loaded successfully!")

# ============================================================
# 5️⃣ Run Recommendation Logic
# ============================================================
if uploaded_file is not None and paper_title.strip() != "":
    journals_df = load_journal_data(uploaded_file)
    st.info(f"✅ Loaded {len(journals_df)} journals for comparison.")

    threshold = 0.5
    top_n = 6

    # Build enriched paper context
    enhanced_paper_text = (
        "This research paper focuses on " + paper_title +
        ". It involves human-machine collaboration, IoT-based control systems, and data-driven decision-making."
    )

    with st.spinner("🔍 Analyzing and finding the most suitable journals..."):
        # Encode paper title
        paper_embedding = model.encode([enhanced_paper_text])
        # Encode journal topics
        journal_embeddings = model.encode(journals_df["combined_text"].tolist(), show_progress_bar=False)

        # Compute cosine similarity
        similarity_scores = cosine_similarity(paper_embedding, journal_embeddings)[0]
        score_list = list(enumerate(similarity_scores))
        sorted_scores = sorted(score_list, key=lambda x: x[1], reverse=True)

        # Filter and rank
        filtered_scores = [(idx, score) for idx, score in sorted_scores if score >= threshold]
        if len(filtered_scores) < top_n:
            filtered_scores = sorted_scores[:top_n]

        top_indices = [idx for idx, _ in filtered_scores]
        top_journals = journals_df.iloc[top_indices]["Journal_Name"].tolist()
        top_scores = [round(score, 3) for _, score in filtered_scores]

        # Deduplicate journals
        unique_recommendations = []
        for j, s in zip(top_journals, top_scores):
            if j not in [r["Journal_Name"] for r in unique_recommendations]:
                unique_recommendations.append({"Journal_Name": j, "Similarity": s})
        unique_recommendations = unique_recommendations[:top_n]

    # ============================================================
    # 6️⃣ Display Results
    # ============================================================
    st.subheader("🎯 Top Recommended Journals:")
    if unique_recommendations:
        results_df = pd.DataFrame(unique_recommendations)
        st.table(results_df)
        st.download_button(
            label="💾 Download Results as CSV",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name="journal_recommendations.csv",
            mime="text/csv",
        )
    else:
        st.warning("No strong matches found. Try lowering the threshold or adjusting the title wording.")
else:
    st.info("⬅️ Please upload a journal sheet and enter a paper title to begin.")

# ============================================================
# 7️⃣ Footer
# ============================================================
st.markdown("---")
st.caption("Developed with 🧠 SPECTER Embeddings for research-level accuracy • Semantic AI Journal Finder")
