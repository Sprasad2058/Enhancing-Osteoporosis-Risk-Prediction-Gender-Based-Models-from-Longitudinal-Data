import pandas as pd
from sklearn.impute import KNNImputer

# Load your dataset
file_path = r'C:\Users\shyam\Downloads\v1medhx\v1medhx.xlsx'
df = pd.read_excel(file_path)

# Step 1: Identify columns with whole numbers and max decimal precision for other columns
whole_number_columns = []
decimal_precisions = {}

for column in df.columns:
    if df[column].dropna().apply(lambda x: x.is_integer() if pd.notna(x) else False).all():
        whole_number_columns.append(column)
    else:
        max_decimal_places = df[column].dropna().apply(lambda x: len(str(x).split(".")[1]) if "." in str(x) else 0).max()
        decimal_precisions[column] = max_decimal_places

# Step 2: Initialize the KNN Imputer and perform imputation on all columns
imputer = KNNImputer(n_neighbors=5)  
imputed_data = imputer.fit_transform(df) 

# Convert the imputed numpy array back to a DataFrame with the original columns
df_imputed = pd.DataFrame(imputed_data, columns=df.columns)

# Step 3: Format Columns
for column in df_imputed.columns:
    if column in whole_number_columns:
        # Convert to integer if it's a whole number column
        df_imputed[column] = df_imputed[column].round().astype(int)
    elif column in decimal_precisions:
        # Round to the specific number of decimal places if applicable
        precision = decimal_precisions[column]
        df_imputed[column] = df_imputed[column].round(precision)

# Display the updated data after KNN imputation with formatting
print("\nData after KNN Imputation with Correct Formatting:")
print(df_imputed.head())

# Save the updated DataFrame back to an Excel file
output_file_path = r'C:\Users\shyam\Downloads\v1medhx\v1medhx_after imputation.xlsx'
df_imputed.to_excel(output_file_path, index=False)

print(f"\nImputed data with formatting preserved saved to {output_file_path}.")
