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
        df = df.drop(axis=1 , columns=['QuizName','QuizClass','FirstName','LastName','StudentID','CustomID'])
        
        # Shuffle the rows
        df = df.sample(n=50, random_state=101).reset_index(drop=True)
        id_series = pd.Series(df.index + 1, name='ID')        
        df = pd.concat([id_series, df], axis=1)
        # Write to CSV with UTF-8 encoding
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        print(f"Successfully converted {excel_file} to {csv_file}")
    except FileNotFoundError:
        print(f"Error: File {excel_file} not found")
    except Exception as e:  
        print(f"Error: {e}")

if __name__ == "__main__":
    
    
    input_file = Path("Documents/quiz-M4_Intensive Physics 2_25_5-full.xlsx")
    output_file = Path("Documents/Intensive_Physics_2.csv")
    
    transform_excel_to_csv(input_file, output_file)