# import pandas as pd
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

# # ============================================================
# # STEP 1: Load Input Files
# # ============================================================

# journals_df = pd.read_excel("available-journal-master_export_25-Sep-2025.xlsx")
# papers_df = pd.read_excel("papersubmission-master_export_25-Sep-2025 (1).xlsx")

# print("✅ Files loaded successfully...")

# # ============================================================
# # STEP 2: Data Cleaning & Preparation
# # ============================================================

# # Replace missing or invalid text with empty strings
# journals_df["Special_Issue_Name"] = journals_df["Special_Issue_Name"].fillna("").astype(str)
# journals_df["Special_Issue_keywords"] = journals_df["Special_Issue_keywords"].fillna("").astype(str)
# journals_df["Journal_Name"] = journals_df["Journal_Name"].fillna("").astype(str)

# # Combine fields to form the full journal topic text
# journals_df["combined_text"] = journals_df["Special_Issue_Name"] + " " + journals_df["Special_Issue_keywords"]

# # Clean up paper titles (remove NaN, numbers, etc.)
# papers_df["Title_of_paper"] = papers_df["Title_of_paper"].fillna("").astype(str)
# papers_df = papers_df[papers_df["Title_of_paper"].str.strip() != ""]  # remove empty rows

# print(f"🧩 Total journals loaded: {len(journals_df)}")
# print(f"🧩 Total paper titles loaded: {len(papers_df)}")

# # ============================================================
# # STEP 3: Load Pretrained Sentence-BERT Model
# # ============================================================

# print("\n🔄 Loading semantic model (this may take a few seconds)...")
# model = SentenceTransformer("all-MiniLM-L6-v2")
# print("✅ Model loaded successfully.\n")

# # ============================================================
# # STEP 4: Generate Embeddings
# # ============================================================

# print("🧠 Encoding paper titles...")
# paper_titles = papers_df["Title_of_paper"].tolist()
# paper_embeddings = model.encode(paper_titles, show_progress_bar=True)

# print("\n🧠 Encoding journal topics...")
# journal_texts = journals_df["combined_text"].tolist()
# journal_embeddings = model.encode(journal_texts, show_progress_bar=True)

# # ============================================================
# # STEP 5: Compute Semantic Similarity
# # ============================================================

# print("\n📊 Computing cosine similarity...")
# similarity_matrix = cosine_similarity(paper_embeddings, journal_embeddings)

# # ============================================================
# # STEP 6: Get Top 6 Matching Journals
# # ============================================================

# top_n = 6
# recommendations = []

# print("\n🔍 Generating top journal recommendations...\n")

# for i, title in enumerate(paper_titles):
#     # Get similarity scores for this paper
#     sim_scores = list(enumerate(similarity_matrix[i]))
#     # Sort descending by similarity
#     sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#     # Pick top N
#     top_indices = [idx for idx, score in sorted_scores[:top_n]]
#     top_journals = journals_df.iloc[top_indices]["Journal_Name"].tolist()
    
#     # Store result
#     recommendations.append({
#         "Title_of_Paper": title,
#         "Suggested_Journals": ", ".join(top_journals)
#     })

# # ============================================================
# # STEP 7: Export Output
# # ============================================================

# output_df = pd.DataFrame(recommendations)
# output_path = "journal_recommendations.xlsx"
# output_df.to_excel(output_path, index=False)

# print(f"✅ Recommendations generated and saved to: {output_path}")
# print(f"📁 Total papers processed: {len(output_df)}")


import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# STEP 1: Load Input Files
# ============================================================

journals_df = pd.read_excel("available-journal-master_export_25-Sep-2025.xlsx")
papers_df = pd.read_excel("papersubmission-master_export_25-Sep-2025 (1).xlsx")

print("✅ Files loaded successfully...")

# ============================================================
# STEP 2: Data Cleaning & Preparation
# ============================================================

# Clean journals data
journals_df["Journal_Name"] = journals_df["Journal_Name"].fillna("").astype(str)
journals_df["Special_Issue_Name"] = journals_df["Special_Issue_Name"].fillna("").astype(str)
journals_df["Special_Issue_keywords"] = journals_df["Special_Issue_keywords"].fillna("").astype(str)

# Weighted combination: prioritize keywords over name (1.5× weight)
journals_df["combined_text"] = (
    journals_df["Special_Issue_keywords"] + " " + journals_df["Special_Issue_keywords"] + " " + journals_df["Special_Issue_Name"]
)

# Clean papers data
papers_df["Title_of_paper"] = papers_df["Title_of_paper"].fillna("").astype(str)
papers_df = papers_df[papers_df["Title_of_paper"].str.strip() != ""]
papers_df.reset_index(drop=True, inplace=True)

print(f"🧩 Total journals loaded: {len(journals_df)}")
print(f"🧩 Total paper titles loaded: {len(papers_df)}")

# ============================================================
# STEP 3: Load Pretrained Sentence-BERT Model
# ============================================================

print("\n🔄 Loading semantic model (this may take a few seconds)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Model loaded successfully.\n")

# ============================================================
# STEP 4: Generate Embeddings
# ============================================================

print("🧠 Encoding paper titles...")
paper_titles = papers_df["Title_of_paper"].tolist()
paper_embeddings = model.encode(paper_titles, show_progress_bar=True)

print("\n🧠 Encoding journal topics...")
journal_texts = journals_df["combined_text"].tolist()
journal_embeddings = model.encode(journal_texts, show_progress_bar=True)

# ============================================================
# STEP 5: Compute Semantic Similarity
# ============================================================

print("\n📊 Computing cosine similarity...")
similarity_matrix = cosine_similarity(paper_embeddings, journal_embeddings)

# ============================================================
# STEP 6: Generate Top 6 Unique & Relevant Journals
# ============================================================

top_n = 6
threshold = 0.5
recommendations = []

print("\n🔍 Generating refined top journal recommendations...\n")

for i, title in enumerate(paper_titles):
    sim_scores = list(enumerate(similarity_matrix[i]))
    sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Filter by similarity threshold
    filtered_scores = [(idx, score) for idx, score in sorted_scores if score >= threshold]
    if len(filtered_scores) < top_n:
        filtered_scores = sorted_scores[:top_n]

    top_indices = [idx for idx, _ in filtered_scores]
    top_journals = journals_df.iloc[top_indices]["Journal_Name"].tolist()

    # Deduplicate journal names while preserving order
    unique_journals = []
    for j in top_journals:
        if j not in unique_journals:
            unique_journals.append(j)

    # Keep only top N unique journals
    unique_journals = unique_journals[:top_n]

    recommendations.append({
        "Title_of_Paper": title,
        "Suggested_Journals": ", ".join(unique_journals)
    })

# ============================================================
# STEP 7: Export Output
# ============================================================

output_df = pd.DataFrame(recommendations)
output_path = "journal_recommendations.xlsx"
output_df.to_excel(output_path, index=False)

print(f"✅ Recommendations generated and saved to: {output_path}")
print(f"📁 Total papers processed: {len(output_df)}")
print("\n🎯 Process completed successfully.")