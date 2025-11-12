# import streamlit as st
# import pandas as pd
# import os
# import json
# import re
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# from groq import Groq
# from dotenv import load_dotenv

# # ============================================================
# # ✅ Load Environment
# # ============================================================
# load_dotenv()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# client = Groq(api_key=GROQ_API_KEY)

# # ============================================================
# # ✅ Local Embedding Model
# # ============================================================
# @st.cache_resource
# def load_local_model():
#     return SentenceTransformer("all-MiniLM-L6-v2")

# embed_model = load_local_model()

# # ============================================================
# # ✅ Streamlit UI
# # ============================================================
# st.title("📘 Journal Recommendation (Ultra-Fast · Single Groq Call)")
# uploaded_file = st.sidebar.file_uploader("Upload Journal Sheet (.xlsx)")
# paper_title = st.sidebar.text_area("Enter Research Paper Title")
# top_n = st.sidebar.slider("Top N Journals", 3, 20, 5)

# # ============================================================
# # ✅ Load Journal Data
# # ============================================================
# @st.cache_data
# def load_journal_data(uploaded_file):
#     df = pd.read_excel(uploaded_file).fillna("")
#     df["combined_text"] = (
#         df["Journal_Name"].astype(str) + ". " +
#         df["Special_Issue_keywords"].astype(str) + ". " +
#         df["Special_Issue_Name"].astype(str)
#     )
#     return df

# # ============================================================
# # ✅ Single Groq Call (Batch Re-Rank)
# # ============================================================
# def groq_batch_score(paper, journal_texts):

#     formatted_list = "\n".join([f"{i+1}. {txt}" for i, txt in enumerate(journal_texts)])

#     prompt = f"""
#     You are an expert in journal-paper topic matching.

#     Rate the relevance of each journal (0 to 1) for the given paper.

#     Paper: "{paper}"

#     Journals:
#     {formatted_list}

#     Return ONLY a JSON array like this:
#     [
#       {{ "index": 1, "score": 0.82 }},
#       {{ "index": 2, "score": 0.15 }},
#       ...
#     ]
#     No explanations. JSON only.
#     """

#     response = client.chat.completions.create(
#         model="llama-3.1-8b-instant",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0,
#         max_tokens=500
#     )

#     text = response.choices[0].message.content.strip()

#     json_str = re.search(r"\[.*\]", text, re.S)
#     if not json_str:
#         return []

#     try:
#         return json.loads(json_str.group(0))
#     except:
#         return []


# # ============================================================
# # ✅ Main Logic
# # ============================================================
# if uploaded_file and paper_title.strip():

#     df = load_journal_data(uploaded_file)

#     # ✅✅✅ DEDUP FIX ADDED — NO OTHER CHANGES ✅✅✅
#     df["keyword_len"] = df["combined_text"].str.len()
#     df = df.sort_values("keyword_len", ascending=False)
#     df = df.drop_duplicates(subset=["Journal_Name"])
#     df = df.drop(columns=["keyword_len"])
#     st.success(f"✅ Loaded {len(df)} unique journals after removing duplicates.")
#     # ✅✅✅ END FIX ✅✅✅

#     # ✅ Local Embeddings Pre-Filter (Fast)
#     paper_emb = embed_model.encode([paper_title])
#     journal_emb = embed_model.encode(df["combined_text"].tolist())

#     sims = cosine_similarity(paper_emb, journal_emb)[0]
#     df["local_sim"] = sims

#     # Pick top 15 for Groq refinement
#     top_candidates = df.sort_values("local_sim", ascending=False).head(15)

#     st.info("✅ Running ONE Groq call to score top journal candidates...")

#     journal_text_list = top_candidates["combined_text"].tolist()
#     results = groq_batch_score(paper_title, journal_text_list)

#     if not results:
#         st.error("❌ Error: Groq response not readable.")
#     else:
#         indexed = []
#         for r in results:
#             idx = r["index"] - 1
#             score = float(r["score"])
#             journal_name = top_candidates.iloc[idx]["Journal_Name"]
#             indexed.append((journal_name, score))

#         final_sorted = sorted(indexed, key=lambda x: x[1], reverse=True)[:top_n]

#         final_df = pd.DataFrame(final_sorted, columns=["Journal_Name", "Relevance"])
#         final_df["Relevance"] = final_df["Relevance"].round(3)

#         st.subheader("🎯 Final Recommended Journals")
#         st.table(final_df)

#         st.download_button(
#             "Download Results",
#             final_df.to_csv(index=False).encode("utf-8"),
#             "journal_results.csv",
#             "text/csv"
#         )

# else:
#     st.info("Upload sheet + enter title to run.")


import streamlit as st
import pandas as pd
import os
import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from dotenv import load_dotenv

# ============================================================
# ✅ Load Environment
# ============================================================
# ✅ Read API key from Streamlit Secrets
groq_api_key = st.secrets["GROQ_API_KEY"]

# ✅ Initialize client
client = Groq(api_key=groq_api_key)

# ============================================================
# ✅ Local Embedding Model
# ============================================================
@st.cache_resource
def load_local_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_local_model()

# ============================================================
# ✅ Streamlit UI
# ============================================================
st.title("📘 Journal Recommendation (Ultra-Fast · Single Groq Call)")
uploaded_file = st.sidebar.file_uploader("Upload Journal Sheet (.xlsx)")
paper_title = st.sidebar.text_area("Enter Research Paper Title")

# ✅ ✅ ADD THIS: Submit Button
submit = st.sidebar.button("Submit")

top_n = st.sidebar.slider("Top N Journals", 3, 20, 5)

# ============================================================
# ✅ Load Journal Data
# ============================================================
@st.cache_data
def load_journal_data(uploaded_file):
    df = pd.read_excel(uploaded_file).fillna("")
    df["combined_text"] = (
        df["Journal_Name"].astype(str) + ". " +
        df["Special_Issue_keywords"].astype(str) + ". " +
        df["Special_Issue_Name"].astype(str)
    )
    return df

# ============================================================
# ✅ Single Groq Call (Batch Re-Rank)
# ============================================================
def groq_batch_score(paper, journal_texts):

    formatted_list = "\n".join([f"{i+1}. {txt}" for i, txt in enumerate(journal_texts)])

    prompt = f"""
    You are an expert in journal-paper topic matching.
    Rate the relevance of each journal (0 to 1) for the given paper.

    Paper: "{paper}"

    Journals:
    {formatted_list}

    Return ONLY a JSON array like this:
    [
      {{ "index": 1, "score": 0.82 }},
      {{ "index": 2, "score": 0.15 }},
      ...
    ]
    No explanations. JSON only.
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=500
    )

    text = response.choices[0].message.content.strip()
    json_str = re.search(r"\[.*\]", text, re.S)
    if not json_str:
        return []

    try:
        return json.loads(json_str.group(0))
    except:
        return []


# ============================================================
# ✅ ✅ Main Logic runs ONLY when Submit is clicked
# ============================================================
if submit:

    if not uploaded_file:
        st.error("❌ Please upload a journal sheet.")
        st.stop()

    if not paper_title.strip():
        st.error("❌ Please enter a research paper title.")
        st.stop()

    df = load_journal_data(uploaded_file)

    # ✅ Dedup fix
    df["keyword_len"] = df["combined_text"].str.len()
    df = df.sort_values("keyword_len", ascending=False)
    df = df.drop_duplicates(subset=["Journal_Name"])
    df = df.drop(columns=["keyword_len"])

    st.success(f"✅ Loaded {len(df)} unique journals after removing duplicates.")

    # ✅ Local Embeddings Pre-Filter (Fast)
    paper_emb = embed_model.encode([paper_title])
    journal_emb = embed_model.encode(df["combined_text"].tolist())
    sims = cosine_similarity(paper_emb, journal_emb)[0]

    df["local_sim"] = sims

    # Pick top 15 for Groq refinement
    top_candidates = df.sort_values("local_sim", ascending=False).head(15)

    st.info("✅ Running ONE Groq call to score top journal candidates...")

    journal_text_list = top_candidates["combined_text"].tolist()
    results = groq_batch_score(paper_title, journal_text_list)

    if not results:
        st.error("❌ Groq returned no results.")
        st.stop()

    indexed = []
    for r in results:
        idx = r["index"] - 1
        score = float(r["score"])
        journal_name = top_candidates.iloc[idx]["Journal_Name"]
        indexed.append((journal_name, score))

    final_sorted = sorted(indexed, key=lambda x: x[1], reverse=True)[:top_n]

    final_df = pd.DataFrame(final_sorted, columns=["Journal_Name", "Relevance"])
    final_df["Relevance"] = final_df["Relevance"].round(3)

    st.subheader("🎯 Final Recommended Journals")
    st.table(final_df)

    st.download_button(
        "Download Results",
        final_df.to_csv(index=False).encode("utf-8"),
        "journal_results.csv",
        "text/csv"
    )
