# VectorShop Performance Metrics

This document provides performance benchmarks and optimization guidelines for VectorShop.

## Search Performance

### Response Time

| Configuration | Average Response Time | 95th Percentile |
|---------------|----------------------|-----------------|
| Demo Mode (Colab CPU) | 0.6 seconds | 0.8 seconds |
| Full System (Colab CPU) | 145 seconds | 185 seconds |
| Full System (Colab L4 GPU) | 12 seconds | 15 seconds |
| Optimized System | 2-3 seconds | 5 seconds |

### Accuracy

| Query Type | Target Product Found | Rank |
|------------|---------------------|------|
| Simple Keyword | 95% | Top 3 |
| Natural Language | 90% | Top 5 |
| Complex Constraints | 85% | Top 10 |

**Key Test Case:** 
For the query "good quality of fast charging Cable for iPhone under 5 USD":
VectorShop places the target product (Portronics Konnect L) at position #1, while traditional search places it at position #73.

## Memory Usage

| Component | RAM Required (CPU) | VRAM Required (GPU) |
|-----------|-------------------|---------------------|
| Full System | 8-10 GB | 12+ GB |
| DeepSeek Model | 4-6 GB | 6-8 GB |
| FAISS Indexes | 200-500 MB | 200-500 MB |
| Data Processing | 2-3 GB | 2-3 GB |
| Demo Mode | 2-3 GB | N/A |

## Scaling Characteristics

VectorShop's performance scales with the following factors:

- **Dataset Size**: Performance is approximately linear with dataset size
- **Query Complexity**: Complex queries with multiple constraints take 20-30% longer
- **Hardware**: GPU acceleration provides 10-12x speedup over CPU-only mode

## Optimization Strategies

To improve performance in production environments:

### 1. Memory Optimizations

- **Quantization**: Using 8-bit quantization reduces memory usage by 50-60%
- **Chunked Processing**: Process large datasets in manageable chunks
- **Lazy Loading**: Only load models when needed

```python
# Example: Memory optimization for DeepSeek model
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    quantization_config=quantization_config,
    device_map="auto"
)
```

### 2. Speed Optimizations

- **Index Caching**: Cache FAISS indexes in memory for faster retrieval
- **Query Preprocessing**: Optimize query processing pipeline
- **Batch Processing**: Process multiple queries in batches when possible

```python
# Example: Creating optimized FAISS index
import faiss

# For datasets < 1M items
index = faiss.IndexFlatIP(dimension)  # Exact, but slower for large datasets

# For datasets > 1M items
nlist = 100  # Number of clusters
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
index.train(embeddings)
```

### 3. Infrastructure Recommendations

- **Minimum Requirements**: 4-core CPU, 8GB RAM for demo mode
- **Recommended**: 8-core CPU, 16GB RAM, NVIDIA GPU with 8GB+ VRAM
- **Production**: Dedicated instance with 16+ cores, 32GB+ RAM, NVIDIA T4/V100 GPU

## Performance Monitoring

In production, monitor the following metrics:

- **Search Latency**: Keep under 1 second for good user experience
- **Memory Usage**: Stay within 80% of available memory
- **GPU Utilization**: Optimize to use 70-80% of GPU
- **Cache Hit Rate**: Aim for 90%+ cache hit rate for common queries

## Conclusion

VectorShop's performance is suitable for small to medium-sized e-commerce catalogs (up to 100,000 products) in its current form. For larger catalogs, additional optimization and infrastructure scaling may be required.

The demo mode provides an excellent balance of performance and resource usage for presentation purposes, while the full system demonstrates the comprehensive capabilities of the semantic search technology.
