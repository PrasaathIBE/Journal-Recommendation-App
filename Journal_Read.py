import pandas as pd

# Load the Excel file
file_path = "available-journal-master_export_25-Sep-2025.xlsx"

# Read the data sheet
df = pd.read_excel(file_path, sheet_name="data")

# Filter conditions
filtered_df = df[
    (df["Journal_Login_Status"].str.strip().str.lower() == "working") &
    (df["Index"].str.strip().str.upper() == "SCIE") &
    (df["SI_Open"].str.strip().str.lower() == "open")
]

# Show result
print("Total journals matching criteria:", len(filtered_df))
print(filtered_df[["Journal_Name", "Special_Issue_Name", "Managing_Guest_Editor", "Index", "SI_Open", "Journal_Login_Status"]].head())

# Optionally export to Excel
filtered_df.to_excel("Filtered_Journals_Working_SCIE_Open.xlsx", index=False)
print("\nFiltered results saved as 'Filtered_Journals_Working_SCIE_Open.xlsx'")
