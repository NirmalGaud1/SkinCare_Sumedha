import streamlit as st
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel 
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from typing import List

# --- CONFIGURATION ---
DATASET_PATH = 'dataset.csv'
MODEL_NAME = 'huawei-noah/TinyBERT_General_4L_312D'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CORE NLP FUNCTIONS (Cached for Performance) ---

@st.cache_resource
def load_model_and_tokenizer():
    """Loads the TinyBERT model and tokenizer once."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    return tokenizer, model

def extract_tinybert_features(texts: List[str], tokenizer, model) -> np.ndarray:
    """Generates TinyBERT [CLS] embeddings for the texts."""
    inputs = tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    ).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    cls_token_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return cls_token_features

def rule_based_extraction(row):
    """Simplified extraction based on keywords (since TinyBERT is not a good JSON generator)."""
    text = row['Raw_Text_Snippet'].lower()
    
    ingredients = []
    if 'ceramide' in text: ingredients.append('Ceramides')
    if 'niacinamide' in text: ingredients.append('Niacinamide')
    if 'retinol' in text or 'retinaldehyde' in text: ingredients.append('Retinoid')
    if not ingredients and 'slugging' not in text and 'skinimalism' not in text: 
         ingredients.append('N/A')
    
    product_types = []
    if 'serum' in text: product_types.append('Serum')
    if 'cleanser' in text or 'wash' in text: product_types.append('Cleanser')
    if 'slugging' in text or 'petroleum' in text: product_types.append('Occlusive Balm')
    if 'device' in text or 'microcurrent' in text: product_types.append('Device')
    if not product_types: product_types.append('N/A')

    category_map = {
        'IDT': 'Ingredient Focus', 'TRT': 'Routine Technique', 'PSI': 'Problem Solver',
        'FSS': 'Sustainable Format', 'CLS': 'Lifestyle Shift', 'FAD': 'Misinformation',
        'UNKNOWN': 'Uncategorized'
    }

    return pd.Series({
        'Trend_Name_Predicted_TBERT': category_map.get(row['Trend_Category_Assigned_TBERT'], 'UNKNOWN'),
        'Associated_Ingredients_Extracted_TBERT': ingredients,
        'Product_Types_Extracted_TBERT': product_types,
        'Key_Benefits_Extracted_TBERT': ['Feature-Based Summary']
    })

# --- MAIN STREAMLIT APP LOGIC ---

def main():
    st.set_page_config(page_title="Skincare Trendspotter POC", layout="wide")
    st.title("🧖‍♀️ Skincare Trendspotter POC (TinyBERT Edition)")
    st.markdown("This application uses **TinyBERT** feature extraction + **Logistic Regression** to categorize emerging skincare trends from your `dataset.csv` file.")

    try:
        data_df = pd.read_csv(DATASET_PATH, on_bad_lines='skip',  # Skip lines that have the wrong number of columns engine='python'       # Use the Python engine, which is more robust than the default C engine)
        st.sidebar.success(f"Successfully loaded {len(data_df)} rows from {DATASET_PATH}.")
        
    except FileNotFoundError:
        st.error(f"FATAL ERROR: `dataset.csv` not found in the current directory.")
        return
    except Exception as e:
        st.error(f"FATAL ERROR during data loading: {e}")
        return

    st.subheader("1. Raw Data Preview")
    st.dataframe(data_df.head(10))

    if st.sidebar.button("Run TinyBERT Trend Analysis"):
        with st.spinner("Loading TinyBERT model and running feature extraction..."):
            tokenizer, model = load_model_and_tokenizer()
            raw_texts = data_df['Raw_Text_Snippet'].tolist()
            X_features = extract_tinybert_features(raw_texts, tokenizer, model)
            st.success("TinyBERT features extracted.")

        with st.spinner("Training Logistic Regression Classifier..."):
            try:
                # Use the synthetic column for training the classifier
                y_labels = data_df['Trend_Category_Assigned'].tolist()
                le = LabelEncoder()
                y_encoded = le.fit_transform(y_labels)
                classifier = LogisticRegression(max_iter=1000, solver='liblinear')
                classifier.fit(X_features, y_encoded)
                y_pred_encoded = classifier.predict(X_features)
                y_pred_labels = le.inverse_transform(y_pred_encoded)
                data_df['Trend_Category_Assigned_TBERT'] = y_pred_labels
                st.success("Trend categories predicted using Logistic Regression.")

            except KeyError:
                st.warning("Missing 'Trend_Category_Assigned' column in CSV for training. Using 'UNKNOWN'.")
                data_df['Trend_Category_Assigned_TBERT'] = 'UNKNOWN'
            except Exception as e:
                st.error(f"Error during classification: {e}")
                return

        with st.spinner("Applying Rule-Based Extraction for structured fields..."):
            extracted_features = data_df.apply(rule_based_extraction, axis=1)
            final_structured_df = pd.concat([data_df, extracted_features], axis=1)
            
            # Clean and select final output columns
            output_columns = ['ID', 'Data_Source_Type', 'Raw_Text_Snippet', 
                              'Trend_Category_Assigned_TBERT', 'Trend_Name_Predicted_TBERT',
                              'Associated_Ingredients_Extracted_TBERT', 
                              'Product_Types_Extracted_TBERT']
            
            final_structured_df = final_structured_df[output_columns].copy()
            st.session_state['results'] = final_structured_df
            st.success("Analysis complete!")


    if 'results' in st.session_state:
        st.subheader("2. Final Structured Trend Analysis")
        st.dataframe(st.session_state['results'])

        # --- 3. Visualization ---
        st.subheader("3. Trend Category Distribution")
        
        # Calculate trend counts
        trend_counts = st.session_state['results']['Trend_Category_Assigned_TBERT'].value_counts().reset_index()
        trend_counts.columns = ['Trend_Category', 'Count']
        
        # Display chart
        st.bar_chart(trend_counts.set_index('Trend_Category'))

if __name__ == "__main__":
    main()
