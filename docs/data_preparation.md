# Data Preparation Guide

This document explains how to prepare your product data for use with VectorShop.

## Data Requirements

VectorShop requires the following product information:

### Required Fields
- **product_id**: Unique identifier for each product
- **product_name**: The full product name
- **category**: Product category (ideally hierarchical)
- **price**: Product price (original and discounted if applicable)

### Recommended Fields
- **description**: Detailed product description
- **features**: Product specifications and features
- **images**: URLs to product images
- **reviews**: Customer reviews of the product
- **rating**: Average product rating

## Data Format

VectorShop expects data in CSV format with the following structure:

```csv
product_id,product_name,category,discounted_price,actual_price,rating,about_product,image_url
B07KY3FNQP,"boAt Bassheads 152 in Ear Wired Earphones with Mic(Active Black)","Electronics|Headphones,Earbuds&Accessories|Headphones|In-Ear",₹449,₹1290,4.1,"Break away from old habits through HD sound via 10mm drivers...",https://example.com/image.jpg
```

## Data Processing Steps

1. Cleaning: Remove invalid characters, normalize text, and handle missing values
2. Price Conversion: Convert prices to a standard currency (USD)
3. Category Structuring: Parse hierarchical categories into a standardized format
4. Text Combination: Create a combined text representation for search
5. Image Processing: Generate image descriptions and embeddings
6. Embedding Generation: Create vector representations of all products


## Example Processing Script

```python
import pandas as pd
from vectorshop.data.preprocessing import create_robust_product_text
from vectorshop.embedding.deepseek_embeddings import DeepSeekEmbeddings

# Load raw data
df = pd.read_csv("data/raw/amazon.csv")

# Clean and preprocess
df['price_usd'] = df['discounted_price'].str.replace('₹', '').str.replace(',', '').astype(float) / 83

# Create combined text representation
df['combined_text_improved'] = df.apply(create_robust_product_text, axis=1)

# Generate embeddings
embeddings_generator = DeepSeekEmbeddings(device="cpu")
embeddings = embeddings_generator.generate_product_embeddings(
    df=df,
    text_column='combined_text_improved',
    output_path="data/processed/embeddings.npy"
)

# Save processed data
df.to_csv("data/processed/amazon_processed.csv", index=False)
```

## Large Datasets

For large datasets (>10,000 products), we recommend:

1. Processing data in chunks (the embedding_tracker module helps with this)
2. Using an incremental update strategy
3. Optimizing memory usage with the provided utility functions

See the vectorshop/embedding/embedding_tracker.py module for implementation details.

