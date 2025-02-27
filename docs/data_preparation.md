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