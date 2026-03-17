import pandas as pd
from pathlib import Path

def transform_excel_to_csv(excel_file, csv_file):
    """
    Transform an Excel file to a CSV file with UTF-8 encoding.
    
    Args:
        excel_file (str): Path to the input Excel file
        csv_file (str): Path to the output CSV file
    """
    try:
        # Read the Excel file
        df = pd.read_excel(excel_file)
        
        # Write to CSV with UTF-8 encoding
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        print(f"Successfully converted {excel_file} to {csv_file}")
    except FileNotFoundError:
        print(f"Error: File {excel_file} not found")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    
    
    input_file = Path("Documents/quiz-M5_Intensive Physics 4_25_5-full.xlsx")
    output_file = Path("Documents/Intensive_Physics_4.csv")
    
    transform_excel_to_csv(input_file, output_file)