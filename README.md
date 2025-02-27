# VectorShop: Semantic Product Search System

VectorShop is a production-ready semantic search system for small-to-medium sized online stores, allowing customers to find products using natural language queries instead of just keywords.

## 🔍 Key Features

- **Natural Language Understanding**: Search for products as you would ask a store associate
- **Hybrid Search Technology**: Combines keyword search, vector similarity, and AI reasoning
- **Cost-Effective**: Uses open-source models to deliver enterprise-grade search capabilities
- **Easy Integration**: Works with Shopify and other e-commerce platforms
- **Multilingual Support**: Handles both English and regional language content

## 🚀 Quick Demo

Try VectorShop with our [interactive demo notebook](demo/VectorShop_Demo.ipynb)!

## 📋 Requirements

- Python 3.9+
- PyTorch 1.9+
- FAISS
- Transformers
- DeepSeek-R1-Distill-Qwen-1.5B

See [requirements.txt](requirements.txt) for the complete list.

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/vectorshop.git
cd vectorshop

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .

## 📖 Documentation

- [System Architecture](docs/system_architecture.md)

- [Data Preparation](docs/data_preparation.md)

- [Performance Metrics](docs/performance_metrics.md)

- [Integration Guide](docs/integration_guide.md)


## 🧪 Example Usage

```bash

from vectorshop.embedding.hybrid_search import HybridSearch

# Initialize search system
search = HybridSearch(
    df=product_data,
    vector_index_path="path/to/vector_index.faiss",
    device="cpu"
)

# Search for products
results = search.search(
    query="wireless earbuds with noise cancellation under 50 USD",
    top_k=5
)

# Display results
print(results[['product_name', 'price_usd', 'score']])
