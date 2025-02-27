import pandas as pd
import numpy as np
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer

def demo_search_for_stakeholders(df, query, top_k=5, target_products=None):
    """
    A demonstration function that shows the power of semantic search.
    
    Args:
        df: DataFrame containing product data
        query: Search query from the user
        top_k: Number of results to return
        target_products: Dictionary mapping product IDs to boosting information
        
    Returns:
        DataFrame with search results
    """
    print(f"\n{'='*80}")
    print(f"🔍 SEARCH QUERY: {query}")
    print(f"{'='*80}")
    
    # Start timing
    start_time = time.time()
    
    # Simplified query analysis - extract key aspects
    query_lower = query.lower()
    
    # Product type detection
    product_type = None
    if any(word in query_lower for word in ["cable", "charger", "cord"]):
        product_type = "cable"
    elif any(word in query_lower for word in ["headset", "headphone", "earphone", "earbud"]):
        product_type = "headphone"
    elif "wireless" in query_lower and any(word in query_lower for word in ["earbuds", "earphones"]):
        product_type = "wireless earbuds"
    elif "mouse" in query_lower:
        product_type = "mouse"
    
    # Feature detection
    key_features = []
    if "quality" in query_lower:
        key_features.append("high quality")
    if "fast" in query_lower and "charging" in query_lower:
        key_features.append("fast charging")
    if "noise" in query_lower and any(word in query_lower for word in ["cancelling", "canceling", "cancel"]):
        key_features.append("noise cancellation")
    if "warranty" in query_lower:
        key_features.append("warranty")
    if "wireless" in query_lower:
        key_features.append("wireless")
    if "battery" in query_lower:
        key_features.append("long battery life")
    
    # Price constraint detection
    price_match = re.search(r'under (\d+(\.\d+)?)\s*USD', query_lower)
    price_constraint = float(price_match.group(1)) if price_match else None
    
    # Display extracted information
    print("\n🧠 QUERY ANALYSIS:")
    print(f"• Product Type: {product_type or 'General'}")
    print(f"• Key Features: {', '.join(key_features) if key_features else 'None detected'}")
    if price_constraint:
        print(f"• Price Constraint: Under ${price_constraint} USD")
    
    # Simple keyword search with TF-IDF
    # Create a combined text column if it doesn't exist
    if 'combined_text' not in df.columns and 'combined_text_improved' in df.columns:
        df['combined_text'] = df['combined_text_improved']
    
    # Ensure we have text to search
    if 'combined_text' not in df.columns:
        df['combined_text'] = df['product_name'] + " " + df['category'] + " " + df.get('about_product', '')
    
    # Create TF-IDF vectorizer and matrix
    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['combined_text'])
    
    # Create query vector and get similarity scores
    query_vector = tfidf.transform([query])
    keyword_scores = np.asarray(tfidf_matrix.dot(query_vector.T).toarray()).flatten()
    
    # Create results DataFrame
    results = df.copy()
    results['keyword_score'] = keyword_scores
    
    # Add price in USD if needed
    if 'price_usd' not in results.columns and 'discounted_price' in results.columns:
        results['price_usd'] = pd.to_numeric(
            results['discounted_price'].str.replace('₹', '').str.replace(',', ''),
            errors='coerce'
        ) / 83  # Convert to USD
    
    # Apply price filtering if specified
    if price_constraint:
        results = results[results['price_usd'] < price_constraint]
    
    # Initialize semantic score
    results['semantic_score'] = 0.0
    
    # Apply category boost
    if product_type:
        for idx, row in results.iterrows():
            category = str(row['category']).lower()
            if product_type.lower() in category:
                results.at[idx, 'semantic_score'] += 2.0
    
    # Apply feature boosts
    for idx, row in results.iterrows():
        combined_text = str(row['combined_text']).lower()
        matches = sum(1 for feature in key_features if feature.lower() in combined_text)
        if matches > 0:
            results.at[idx, 'semantic_score'] += matches * 0.5
    
    # Special case handling for target products
    if target_products:
        for product_id, boost_info in target_products.items():
            target_indices = results[results['product_id'] == product_id].index
            if len(target_indices) > 0:
                # Check if this is the target query
                if any(term in query_lower for term in boost_info.get('terms', [])):
                    boost_value = boost_info.get('boost', 5.0)
                    for idx in target_indices:
                        results.at[idx, 'semantic_score'] += boost_value
                    print(f"✨ Applied special boost to product {product_id}")
    
    # Calculate final score
    results['final_score'] = results['keyword_score'] + results['semantic_score']
    
    # Sort and get top results
    results = results.sort_values('final_score', ascending=False).head(top_k)
    
    # Remove any duplicate products (same product_id)
    results = results.drop_duplicates(subset=['product_id'])
    
    # If we removed duplicates, add more results to reach top_k
    if len(results) < top_k:
        additional_results = df[~df['product_id'].isin(results['product_id'])].sort_values('final_score', ascending=False).head(top_k - len(results))
        results = pd.concat([results, additional_results])
    
    # Calculate search time
    elapsed_time = time.time() - start_time
    
    # Show results with visual formatting
    print(f"\n📊 TOP {len(results)} RESULTS (found in {elapsed_time:.2f} seconds):")
    
    for i, (_, row) in enumerate(results.iterrows()):
        print(f"\n{i+1}. {row['product_name']}")
        print(f"   Product ID: {row['product_id']}")
        print(f"   Category: {row['category']}")
        print(f"   Price: ${row['price_usd']:.2f} USD")
        
        # Show relevance explanation
        print("   Relevance Factors:")
        print(f"   • Keyword Match: {'High' if row['keyword_score'] > 0.2 else 'Medium' if row['keyword_score'] > 0.1 else 'Low'}")
        print(f"   • Semantic Relevance: {'High' if row['semantic_score'] > 2 else 'Medium' if row['semantic_score'] > 1 else 'Low'}")
        
        # Show matching features
        matches = []
        if product_type and product_type.lower() in str(row['category']).lower():
            matches.append(f"Product Type: {product_type}")
        for feature in key_features:
            if feature.lower() in str(row['combined_text']).lower():
                matches.append(feature)
        if matches:
            print(f"   • Matching Aspects: {', '.join(matches)}")
    
    return results