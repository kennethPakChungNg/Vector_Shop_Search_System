# VectorShop Integration Guide

This document provides instructions for integrating VectorShop with popular e-commerce platforms like Shopify.

## Integration Methods

VectorShop can be integrated with e-commerce platforms in several ways:

1. **API Integration**: Direct REST API calls to a VectorShop server
2. **Shopify App**: Custom app installable through the Shopify App Store
3. **JavaScript Widget**: Embeddable search widget for any website

## Shopify Integration

### Option 1: API-based Integration

For Shopify stores, the simplest integration method is using the VectorShop REST API:

1. **Deploy VectorShop API**: Host the VectorShop API on your preferred server or cloud provider
2. **Create App Bridge Integration**: Use Shopify's App Bridge to connect your store
3. **Add Search Interface**: Implement a custom search UI in your Shopify theme

### Example API request:

```json
GET /api/search?q=wireless+earbuds+with+noise+cancellation+under+50+USD

Response:
{
  "results": [
    {
      "product_id": "B09PL79D2X",
      "product_name": "boAt Airdopes 181 in-Ear True Wireless Earbuds with ENx Tech",
      "price_usd": 19.25,
      "score": 0.92,
      "url": "https://example-store.com/products/B09PL79D2X"
    },
    ...
  ],
  "query_analysis": {
    "product_type": "earbuds",
    "features": ["wireless", "noise cancellation"],
    "price_constraint": 50
  }
}
```

### Option 2: Shopify App

For a more integrated experience, you can create a dedicated Shopify App:

1. **Create App in Shopify Partners Portal**
2. **Implement OAuth Flow**: For secure authentication
3. **Add Product Sync**: To keep product data up-to-date
4. **Override Default Search**: Replace Shopify's default search with VectorShop

### Option 3: Theme Integration

For direct theme integration:

1. **Add VectorShop JS Client**: Include the VectorShop JavaScript client
2. **Modify Search Form**: Update the search form to use VectorShop
3. **Style Results**: Customize the appearance of search results

```html
<!-- Example search form integration -->
<form id="vectorshop-search">
  <input type="text" id="vectorshop-query" placeholder="Search products..." />
  <button type="submit">Search</button>
</form>

<div id="vectorshop-results"></div>

<script src="https://cdn.example.com/vectorshop-client.js"></script>
<script>
  const vectorshop = new VectorShopClient({
    apiKey: 'YOUR_API_KEY',
    apiUrl: 'https://api.example.com/search'
  });

  document.getElementById('vectorshop-search').addEventListener('submit', function(e) {
    e.preventDefault();
    const query = document.getElementById('vectorshop-query').value;
    vectorshop.search(query).then(results => {
      const resultsContainer = document.getElementById('vectorshop-results');
      // Render results...
    });
  });
</script>
```

## Other E-commerce Platforms

### WooCommerce

For WordPress stores using WooCommerce:

1. **Install VectorShop WP Plugin**: Upload and activate the plugin
2. **Configure API Connection**: Enter your VectorShop API credentials
3. **Customize Settings**: Adjust settings to match your store's needs

### Custom Platforms

For custom e-commerce platforms:

1. **API Integration**: Use direct API calls
2. **Embed Search Widget**: Use the JavaScript widget
3. **Direct Library Integration**: Import the VectorShop Python library

## Implementation Roadmap

1. **Development Phase (1-2 weeks)**: Set up API server and search endpoints
2. **Testing Phase (1 week)**: Test integration and optimize performance
3. **Deployment Phase (1 day)**: Deploy to production
4. **Monitoring Phase (ongoing)**: Monitor performance and user feedback

## Support

For integration assistance, please contact ngchungpak@gmail.com.


