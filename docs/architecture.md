# VectorShop System Architecture

VectorShop uses a hybrid search approach that combines multiple search technologies to deliver superior product discovery capabilities. This document explains the technical architecture of the system.

## System Overview

VectorShop is built on three complementary search technologies:

1. **BM25 Search**: Traditional keyword-based retrieval for precise matching
2. **Vector Search**: Semantic embedding search for conceptual understanding
3. **AI Reranking**: Deep language understanding for complex intent comprehension

These components work together to provide a comprehensive search experience that understands natural language queries.

## Components

### 1. Data Preprocessing

Before searching can begin, product data undergoes extensive preprocessing:

- **Text Cleaning**: Normalization, stop word removal, and handling of special characters
- **Category Parsing**: Extraction of hierarchical category information
- **Image Analysis**: Generation of image descriptions using BLIP2
- **Embedding Generation**: Creation of vector representations using DeepSeek models

### 2. Query Processing

When a user enters a search query:

- **Query Analysis**: The system extracts:
  - Product type (e.g., headphones, cables)
  - Feature requirements (e.g., noise cancellation, fast charging)
  - Constraints (e.g., price limits)
  - Special handling flags

### 3. Search Pipeline

The search process follows these steps:

1. **BM25 Search**: Retrieves products matching keywords in the query
2. **Vector Search**: Finds semantically similar products using FAISS
3. **Result Merging**: Combines and normalizes scores from both approaches
4. **Boosting**: Applies category, feature, and special product boosts
5. **Reranking**: Uses DeepSeek-R1-Distill-Qwen-1.5B to evaluate relevance
6. **Final Ranking**: Produces the final ordered list of products

### 4. Technical Stack

VectorShop is built on the following technologies:

- **Python**: Core language for all components
- **PyTorch**: Deep learning framework for AI models
- **FAISS**: Vector similarity search from Facebook AI
- **DeepSeek-R1-Distill-Qwen-1.5B**: Foundation model for semantic understanding
- **Pandas**: Data manipulation and processing
- **NLTK/scikit-learn**: Text processing and BM25 implementation

## Memory and Performance Considerations

The system is designed to be efficient while maintaining high-quality results:

- **Chunked Processing**: Large datasets are processed in manageable chunks
- **Incremental Updates**: Only new or modified products need reprocessing
- **Lazy Loading**: Models are loaded only when needed
- **Memory Management**: GPU memory is carefully managed for efficiency

## Integration Interfaces

VectorShop provides multiple integration options:

- **Python API**: Direct integration with Python applications
- **REST API**: HTTP endpoints for web and mobile applications
- **Shopify App**: Dedicated integration for Shopify stores

## Future Expansion

The architecture is designed for easy extension with:

- **Personalization**: User preference integration
- **Multi-language Support**: Additional language capabilities
- **Recommendation Engine**: Product recommendation features