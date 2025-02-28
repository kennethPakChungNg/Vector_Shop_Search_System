import pandas as pd
import numpy as np
import faiss
import os
from typing import Tuple, Optional
from vectorshop.embedding.deepseek_embeddings import DeepSeekEmbeddings

def generate_essential_files(
    data_path: str, 
    output_dir: str, 
    text_column: str = 'combined_text_improved',
    device: str = "cpu"
) -> Tuple[np.ndarray, faiss.Index]:
    """Generate embeddings and FAISS index from raw data.
    
    Args:
        data_path: Path to the dataset CSV
        output_dir: Directory to save generated files
        text_column: Column containing text to embed
        device: Device to use ('cpu' or 'cuda')
        
    Returns:
        Tuple of (embeddings array, FAISS index)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    
    print(f"Generating embeddings for {len(df)} products...")
    embeddings_generator = DeepSeekEmbeddings(device=device)
    embeddings = embeddings_generator.generate_product_embeddings(
        df=df,
        text_column=text_column,
        output_path=f"{output_dir}/embeddings.npy"
    )
    
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    # Save the index
    index_path = f"{output_dir}/vector_index.faiss"
    faiss.write_index(index, index_path)
    
    print(f"Files generated successfully in {output_dir}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"FAISS index size: {index.ntotal} vectors")
    
    return embeddings, index