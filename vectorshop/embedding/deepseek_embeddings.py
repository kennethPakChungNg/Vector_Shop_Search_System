"""
DeepSeek embeddings generator for e-commerce search.
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from typing import List, Union, Dict, Any, Optional
import os
import time
from tqdm import tqdm
import gc

class DeepSeekEmbeddings:
    """
    Generate embeddings using DeepSeek models for semantic search.
    """
    
    def __init__(self, model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", device="cpu"):
        """
        Initialize the DeepSeek embeddings generator.
        
        Args:
            model_name: Name of the DeepSeek model to use
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None
        self.embedding_dim = 2048  # Default, will be updated after first encoding
        
    def load_model(self):
        """Load the DeepSeek model and tokenizer."""
        if self._model is None:
            print(f"Loading {self.model_name} for embeddings...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self._model = AutoModel.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            ).to(self.device)
            self._model.eval()
            print(f"Model loaded successfully on {self.device}")
            # Apply memory optimization
            self.optimize_memory()
        return self._model, self._tokenizer
    
    def ensure_tensor_device(self, tensor_or_dict: Any, target_device: Optional[str] = None) -> Any:
        """
        Ensure all tensors in the input are on the same device.
        
        Args:
            tensor_or_dict: A tensor, dictionary of tensors, or other data structure
            target_device: Target device to move tensors to (defaults to self.device)
            
        Returns:
            The input with all tensors moved to the same device
        """
        device = target_device or self.device
        
        # Handle tensor case
        if isinstance(tensor_or_dict, torch.Tensor):
            return tensor_or_dict.to(device)
        
        # Handle dictionary case (like tokenizer outputs)
        elif isinstance(tensor_or_dict, dict):
            return {
                k: self.ensure_tensor_device(v, device) 
                for k, v in tensor_or_dict.items()
            }
        
        # Handle list/tuple case
        elif isinstance(tensor_or_dict, (list, tuple)):
            return type(tensor_or_dict)(
                self.ensure_tensor_device(x, device) for x in tensor_or_dict
            )
        
        # Non-tensor types are returned as is
        return tensor_or_dict
    
    def optimize_memory(self):
        """Apply memory optimizations to reduce memory usage."""
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def free_memory(self):
        """Free up memory by clearing caches."""
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _mean_pooling(self, model_output, attention_mask):
        """
        Perform mean pooling on token embeddings.
        
        Args:
            model_output: Model output containing hidden states
            attention_mask: Attention mask from tokenizer
            
        Returns:
            Mean-pooled embeddings
        """
        # First element of model_output contains all token embeddings
        token_embeddings = model_output.last_hidden_state
        
        # Mask padding tokens
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sum all token embeddings and divide by the number of tokens
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Return mean-pooled embeddings
        return sum_embeddings / sum_mask
    
    def encode(self, texts: Union[str, List[str]], batch_size=8, normalize=True) -> np.ndarray:
        """
        Generate embeddings for the provided texts.
        
        Args:
            texts: Single text or list of texts
            batch_size: Number of texts to process in a batch
            normalize: Whether to normalize embeddings to unit length
            
        Returns:
            Numpy array of embeddings
        """
        # Load model if not already loaded
        model, tokenizer = self.load_model()
        
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        # Initialize array to store embeddings
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:min(i+batch_size, len(texts))]
            
            try:
                # Tokenize batch
                encoded_input = tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=512, 
                    return_tensors='pt'
                )
                
                # Ensure tensors are on the correct device
                encoded_input = self.ensure_tensor_device(encoded_input)
                
                # Generate embeddings
                with torch.no_grad():
                    model_output = model(**encoded_input)
                
                # Perform mean pooling
                batch_embeddings = self._mean_pooling(
                    model_output, 
                    encoded_input['attention_mask']
                ).cpu().numpy()
                
                # Update embedding_dim based on actual output
                if i == 0:
                    self.embedding_dim = batch_embeddings.shape[1]
                
                # Normalize if requested
                if normalize:
                    batch_embeddings = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                
                embeddings.append(batch_embeddings)
                
                # Clean up after each batch
                self.optimize_memory()
                
            except Exception as e:
                print(f"Error encoding batch {i//batch_size}: {e}")
                # Return empty embeddings for this batch
                batch_embeddings = np.zeros((len(batch_texts), self.embedding_dim))
                embeddings.append(batch_embeddings)
        
        # Concatenate all batch embeddings
        all_embeddings = np.vstack(embeddings)
        
        return all_embeddings
    
    def generate_product_embeddings(self, df: pd.DataFrame, text_column='combined_text_improved', 
                                   output_path=None, batch_size=8):
        """
        Generate embeddings for product descriptions and save to file.
        
        Args:
            df: DataFrame containing product data
            text_column: Column containing text to embed
            output_path: Path to save embeddings (optional)
            batch_size: Number of texts to process in a batch
            
        Returns:
            Numpy array of embeddings
        """
        # Ensure text column exists
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        # Get text data
        texts = df[text_column].fillna("").tolist()
        
        print(f"Generating embeddings for {len(texts)} products...")
        start_time = time.time()
        
        # Generate embeddings with progress bar
        embeddings = self.encode(texts, batch_size=batch_size)
        
        elapsed_time = time.time() - start_time
        print(f"Embeddings generated in {elapsed_time:.2f} seconds")
        
        # Save embeddings if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.save(output_path, embeddings)
            print(f"Embeddings saved to {output_path}")
        
        # Final memory cleanup
        self.free_memory()
        
        return embeddings

def create_product_text(product_row, include_image_desc=True):
    """
    Create a comprehensive text representation of a product for embedding.
    
    Args:
        product_row: Row from the product DataFrame
        include_image_desc: Whether to include image description
        
    Returns:
        String containing formatted product information
    """
    parts = []
    
    # Add product name (most important)
    if 'product_name' in product_row and not pd.isna(product_row['product_name']):
        parts.append(f"Product: {product_row['product_name']}")
    
    # Add category with hierarchy
    if 'category' in product_row and not pd.isna(product_row['category']):
        category = product_row['category']
        # Handle different category separators
        if '|' in category:
            category_parts = category.split('|')
        elif '>' in category:
            category_parts = category.split('>')
        else:
            category_parts = [category]
        
        # Add each category level with appropriate weight
        parts.append(f"Category: {' > '.join(category_parts)}")
        
        # Add primary category separately for emphasis
        if len(category_parts) > 0:
            parts.append(f"Primary Category: {category_parts[0]}")
        
        # Add sub-category separately for emphasis
        if len(category_parts) > 1:
            parts.append(f"Sub-Category: {category_parts[1]}")
    
    # Add product description
    if 'about_product' in product_row and not pd.isna(product_row['about_product']):
        parts.append(f"Description: {product_row['about_product']}")
    
    # Add key attributes
    attrs = []
    if 'rating' in product_row and not pd.isna(product_row['rating']):
        rating = float(product_row['rating']) if isinstance(product_row['rating'], str) else product_row['rating']
        if rating >= 4.0:
            attrs.append("high rating")
        attrs.append(f"rating {rating}")
    
    # Add price information
    if 'discounted_price' in product_row and not pd.isna(product_row['discounted_price']):
        price_str = str(product_row['discounted_price']).replace('₹', '').replace(',', '')
        try:
            price_inr = float(price_str)
            price_usd = price_inr / 83  # Convert to USD
            parts.append(f"Price: {price_usd:.2f} USD")
        except:
            pass
    
    # Add review content if available
    if 'review_content' in product_row and not pd.isna(product_row['review_content']):
        parts.append(f"Reviews: {product_row['review_content']}")
    
    # Add image description if requested and available
    if include_image_desc and 'image_desc' in product_row and not pd.isna(product_row['image_desc']):
        parts.append(f"Image: {product_row['image_desc']}")
    
    # Join all parts with line breaks for better tokenization
    return "\n".join(parts)