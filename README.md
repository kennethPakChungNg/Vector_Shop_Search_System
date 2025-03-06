# VectorShop: Semantic Product Search System

VectorShop is a production-ready semantic search system for small-to-medium sized online stores, allowing customers to find products using natural language queries instead of just keywords.

## 🔍 Key Features

- **Natural Language Understanding**: Search for products as you would ask a store associate
- **Hybrid Search Technology**: Combines keyword search, vector similarity, and AI reasoning
- **Cost-Effective**: Uses open-source models to deliver enterprise-grade search capabilities
- **Easy Integration**: Works with Shopify and other e-commerce platforms
- **Multilingual Support**: Handles both English and regional language content

## 🚀 Quick Demo

Try VectorShop with my [interactive demo notebook](demo/VectorShop_Demo.ipynb)!

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
```

## 📖 Documentation

- [System Architecture](docs/architecture.md)

- [Data Preparation](docs/data_preparation.md)

- [Performance Metrics](docs/performance.md)

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
```

## 📊 Business Impact

- Increased Conversions: Customers find exactly what they're looking for
- Reduced Bounce Rates: Fewer failed searches and abandoned sessions
- Enhanced Customer Experience: Natural interaction with product catalog
- Competitive Advantage: Enterprise-level search capabilities at SMB cost

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Contributors

- [@kennethPakChungNg](https://github.com/kennethPakChungNg)

## 🙏 Acknowledgments

- This project utilizes the DeepSeek-R1-Distill model from DeepSeek AI
- Amazon product dataset from Kaggle

## Workflow Options

VectorShop offers two distinct workflows for different use cases:

### 1. Full System Implementation

The complete implementation located in `vectorshop/notebooks/02_VectorShop_Full_Process.ipynb` provides a production-grade semantic search system with:

- **Image Processing**: Uses BLIP2 for image description generation
- **Advanced Embeddings**: Utilizes DeepSeek and CLIP models for text and image embeddings
- **Hybrid Search**: Combines BM25 keyword search with FAISS vector similarity search
- **AI-Powered Reranking**: Applies DeepSeek reasoning for final result scoring
- **Comprehensive Boosting**: Uses multiple factors for sophisticated result ranking

**Performance Characteristics:**
- Search time: 145-185 seconds per query
- Memory usage: High (~4-8GB RAM depending on configuration)
- GPU acceleration: Recommended for production use
- Result quality: Highest semantic understanding

**Best For:** Production deployments, development and customization, and comprehensive system testing.

### 2. Demo System Implementation

The streamlined implementation located in `demo/VectorShop_Demo.ipynb` provides a fast, presentation-ready search system with:

- **TF-IDF Vectorization**: Simple keyword matching without embeddings
- **Pattern-Based Query Analysis**: Uses regex instead of AI for feature detection
- **Lightweight Boosting**: Basic category and feature boosting for relevant results
- **Fast Execution**: Returns results in under 1 second
- **Guaranteed Results**: Ensures high-quality results for demonstration queries

**Performance Characteristics:**
- Search time: ~0.6 seconds per query
- Memory usage: Low (~500MB RAM)
- GPU acceleration: Not required
- Result quality: Good for common e-commerce queries

**Best For:** Stakeholder presentations, quick demonstrations, and environments without GPU access.

### Relationship Between the Two Workflows

The demo implementation is derived from the full system but optimized for speed and reliability. While it doesn't use the advanced AI components of the full system, it maintains the same core concepts:

1. **Query Analysis**: Both extract product types, features, and constraints
2. **Relevance Scoring**: Both calculate scores based on text similarity and product attributes
3. **Result Presentation**: Both provide detailed explanations of why products match

The demo version is not a replacement for the full system, but rather a companion that makes the same concepts accessible in situations where speed and reliability are crucial.



