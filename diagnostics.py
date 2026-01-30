# Diagnostic: check for non-numeric columns in feature_cols
# Run this cell in Databricks BEFORE Step 1.2 of Book 9
print(f"Total feature columns: {len(feature_cols)}")
print(f"\nNon-numeric columns in feature_cols:")
found = False
for col in feature_cols:
    dtype = df_pandas[col].dtype
    if not pd.api.types.is_numeric_dtype(df_pandas[col]):
        print(f"  {col}: {dtype}")
        found = True
if not found:
    print("  (none)")

print(f"\nAll dtypes in feature_cols:")
print(df_pandas[feature_cols].dtypes.value_counts())
