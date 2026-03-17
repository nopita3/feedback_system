import pandas as pd
import json
from pathlib import Path

def read_token_log(file_path):
    """
    Read Token_GeminiAPI_usage_log.txt and create a dataframe with token analytics.
    Parses JSON-formatted log entries.
    """
    data = []
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            
            try:
                # Parse JSON log entry
                log_entry = json.loads(line)
                
                # Extract timestamp (first key)
                timestamp = list(log_entry.keys())[0]
                
                # Get the model data (usually gemini-3.1-flash-lite-preview)
                model_data = log_entry.get(timestamp, {})
                if isinstance(model_data, dict):
                    # Get first model's data
                    model_key = list(model_data.keys())[0]
                    tokens = model_data[model_key]
                    
                    row = {
                        'timestamp': timestamp,
                        'model': model_key,
                        'input_tokens': tokens.get('input_tokens', 0),
                        'output_tokens': tokens.get('output_tokens', 0),
                        'total_tokens': tokens.get('total_tokens', 0),
                        'process': list(log_entry.keys())[-1]
                    }
                    data.append(row)
            except (json.JSONDecodeError, KeyError, IndexError):
                # Skip lines that can't be parsed
                continue
    
    df = pd.DataFrame(data)
    return df


# Usage
log_file = Path('Token_GeminiAPI_usage_log.txt')
df = read_token_log(log_file)
print(df)
work_type = df['process'].unique()
print(f"\nWork Types: {work_type}")

# Create visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Group by work_type and calculate totals
work_type_summary = df.groupby('process').agg({
    'input_tokens': 'sum',
    'output_tokens': 'sum',
    'total_tokens': 'sum'
}).reset_index()

print("\nToken Summary by Work Type:")
print(work_type_summary)

# Create dataframe for grouped bar chart
chart_data = []
for _, row in work_type_summary.iterrows():
    process = row['process']
    chart_data.append({'Token Type': 'Total Tokens', 'Count': row['total_tokens'], 'Work Type': process})
    chart_data.append({'Token Type': 'Input Tokens', 'Count': row['input_tokens'], 'Work Type': process})
    chart_data.append({'Token Type': 'Output Tokens', 'Count': row['output_tokens'], 'Work Type': process})

chart_df = pd.DataFrame(chart_data)

# Create figure and plot with hue for work_type
plt.figure(figsize=(12, 6))
ax = sns.barplot(data=chart_df, x='Token Type', y='Count', hue='Work Type', palette='Set2')

# Annotate bars with values
for bar in ax.patches:
    height = bar.get_height()
    if height > 0:
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.title('Token Usage Summary by Work Type', fontsize=14, fontweight='bold')
plt.ylabel('Token Count', fontsize=12)
plt.xlabel('Token Type', fontsize=12)
plt.legend(title='Work Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('Token_usage_summary.png', dpi=300)
plt.show()

# Overall summary
print("\nOverall Token Summary:")
total_input = df['input_tokens'].sum()
total_output = df['output_tokens'].sum()
total_all = df['total_tokens'].sum()
print(f"Total Tokens: {total_all:,}")
print(f"Input Tokens: {total_input:,}")
print(f"Output Tokens: {total_output:,}")
