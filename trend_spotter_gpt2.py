import pandas as pd
import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from typing import Dict, Any, List

# --- Configuration ---
DATASET_PATH = 'dataset.csv'
MODEL_NAME = 'gpt2' # Using the smallest base GPT-2 model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 1. Model and Tokenizer Initialization ---
# Load the pre-trained GPT-2 model and tokenizer
# We load the base GPT2Model to get the text embeddings (features)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2Model.from_pretrained(MODEL_NAME).to(DEVICE)

# GPT-2 does not have a default padding token, but one is needed for batch processing.
# We set the end-of-sequence token as the padding token.
tokenizer.padding_side = "left" # Crucial for GPT-2 classification/feature extraction
tokenizer.pad_token = tokenizer.eos_token

# --- 2. Feature Extraction Function (The GPT-2 Logic) ---
def extract_gpt2_features(texts: List[str]) -> np.ndarray:
    """
    Tokenizes a list of text snippets and extracts the final token's hidden state
    (the context vector) from the GPT-2 model. This vector is used as the feature.
    """
    # 1. Tokenize the input texts
    # Pad sequences to the left and truncate to the maximum length (1024 for base GPT-2)
    inputs = tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=1024
    ).to(DEVICE)
    
    # 2. Get model outputs (embeddings)
    with torch.no_grad():
        outputs = model(**inputs)
        
    # 3. Extract the final hidden state (embeddings)
    # Since GPT-2 processes left-to-right, the last non-padding token's hidden state 
    # (the last token's position in the padded sequence) best summarizes the entire input.
    
    # Get the hidden state for the last token of each sequence
    # hidden_states shape: (batch_size, sequence_length, hidden_size)
    hidden_states = outputs.last_hidden_state
    
    # Find the position of the last non-padding token (which is the last token overall since we pad on the left)
    # We take the features from the token *before* the left-padded tokens.
    # The last column of the hidden_states tensor corresponds to the *final* token's representation.
    
    # Simply extract the hidden state of the last token in the sequence dimension (axis=1, index=-1)
    # This works because we padded on the left.
    final_token_features = hidden_states[:, -1, :].cpu().numpy()
    
    return final_token_features

# --- 3. Main Processing Logic ---
def process_trends_poc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses GPT-2 features and a Logistic Regression head to predict the trend category.
    
    NOTE: For the extraction columns (Ingredients, Benefits, etc.), we will use 
    simple keyword matching as a non-LLM, rule-based approach for this POC, 
    as GPT-2 is primarily a text generator/feature extractor, not a structured
    instruction-follower like Gemini/GPT-4.
    """
    
    # --- Part 1: Feature Extraction ---
    raw_texts = df['Raw_Text_Snippet'].tolist()
    print(f"\n--- 1. Extracting Features using GPT-2 Model on {len(raw_texts)} rows ---")
    
    # This extracts the (200, 768) feature matrix (768 is GPT-2 base hidden size)
    X_features = extract_gpt2_features(raw_texts)
    print(f"Feature matrix shape: {X_features.shape}")
    
    # --- Part 2: Simplified Classification (Training a classifier) ---
    # Since this is a POC on synthetic data, we'll simulate the classification step.
    # In a *real* scenario, you would need labeled data to train the classifier.
    
    # 2a. Target Labels (Simulated)
    # We will use the *synthetic* column 'Trend_Category_Assigned' as the 'True' label 
    # to train a simple classifier. This requires that column to exist in the loaded CSV.
    try:
        y_labels = df['Trend_Category_Assigned'].tolist()
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_labels)
        
    except KeyError:
        print("\n!! ERROR: Missing 'Trend_Category_Assigned' column in CSV. Cannot run classification.")
        print("Falling back to a single prediction category.")
        df['Trend_Category_Assigned_GPT2'] = 'IDT_SIMULATED' # Fallback
        return df

    # 2b. Train a Simple Classifier (Logistic Regression)
    # Using the features (X) and the existing categories (y)
    # In a real POC, you'd split this into train/test sets.
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_features, y_encoded)
    
    # 2c. Predict Categories
    y_pred_encoded = classifier.predict(X_features)
    y_pred_labels = le.inverse_transform(y_pred_encoded)
    
    df['Trend_Category_Assigned_GPT2'] = y_pred_labels
    
    print(f"Classification completed. Categories predicted: {np.unique(y_pred_labels)}")

    # --- Part 3: Extraction (Rule-Based Fallback) ---
    # Since GPT-2 is poor at constrained JSON output, we use basic rule matching
    # to populate the structured extraction columns for the POC.
    
    def rule_based_extraction(row):
        text = row['Raw_Text_Snippet'].lower()
        
        # Simple Ingredient Extraction
        ingredients = []
        if 'ceramide' in text: ingredients.append('Ceramides')
        if 'niacinamide' in text: ingredients.append('Niacinamide')
        if 'retinol' in text or 'retinaldehyde' in text: ingredients.append('Retinoid')
        if not ingredients and 'slugging' not in text and 'skinimalism' not in text: 
             ingredients.append('N/A')
        
        # Simple Product/Technique Extraction
        product_types = []
        if 'serum' in text: product_types.append('Serum')
        if 'cleanser' in text or 'wash' in text: product_types.append('Cleanser')
        if 'slugging' in text or 'petroleum' in text: product_types.append('Occlusive Balm')
        if 'device' in text or 'microcurrent' in text: product_types.append('Device')
        if not product_types: product_types.append('N/A')

        # Note: 'Trend_Name_Predicted' and 'Key_Benefits_Extracted' would require more complex NLP 
        # (like zero-shot classification or fine-tuning, which is beyond simple feature extraction). 
        # For the POC, we link the name to the predicted category:
        
        category_map = {
            'IDT': 'Ingredient Focus', 'TRT': 'Routine Technique', 'PSI': 'Problem Solver',
            'FSS': 'Sustainable Format', 'CLS': 'Lifestyle Shift', 'FAD': 'Misinformation'
        }

        return pd.Series({
            'Trend_Name_Predicted_GPT2': category_map.get(row['Trend_Category_Assigned_GPT2'], 'UNKNOWN'),
            'Associated_Ingredients_Extracted_GPT2': ingredients,
            'Product_Types_Extracted_GPT2': product_types,
            'Key_Benefits_Extracted_GPT2': ['Feature-Based Summary']
        })

    print("\n--- 3. Populating Structured Fields with Rule-Based Fallback ---")
    extracted_features = df.apply(rule_based_extraction, axis=1)
    
    # Merge the new columns back to the DataFrame
    df = pd.concat([df, extracted_features], axis=1)
    
    return df.drop(columns=['Trend_Category_Assigned']) # Drop the synthetic true column for final output


# --- Execution ---
if __name__ == "__main__":
    print(f"--- Skincare Trendspotting POC: GPT-2 Feature Structuring Initiated ---")
    
    # 1. Load Data
    try:
        # Load the CSV. We rely on the 'Trend_Category_Assigned' column for simulated training.
        data_df = pd.read_csv(DATASET_PATH)
        print(f"Successfully loaded {len(data_df)} rows from {DATASET_PATH}.")
        
    except FileNotFoundError:
        print(f"!! ERROR: File not found at {DATASET_PATH}. Please ensure it is in the repository.")
        exit()
    except Exception as e:
        print(f"!! ERROR during data loading: {e}. Check CSV formatting.")
        exit()

    # 2. Process Data with GPT-2 Features
    # NOTE: It's HIGHLY recommended to run the full 200 rows for proper classification training.
    # The extraction part (Part 3) is a simplified placeholder.
    
    final_structured_df = process_trends_poc(data_df)

    # 3. Save Results
    OUTPUT_FILE = 'structured_trends_gpt2_poc_output.csv'
    
    # Select and reorder the relevant output columns for a cleaner final result
    output_columns = ['ID', 'Data_Source_Type', 'Raw_Text_Snippet', 
                      'Trend_Category_Assigned_GPT2', 'Trend_Name_Predicted_GPT2',
                      'Associated_Ingredients_Extracted_GPT2', 'Product_Types_Extracted_GPT2']
                      
    final_structured_df = final_structured_df[output_columns]
    
    final_structured_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n--- POC Complete ---")
    print(f"Structured results saved to: {OUTPUT_FILE}")
    print("\nNext step: Analyze the structured output and build the prioritization model.")
