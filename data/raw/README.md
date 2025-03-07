# Raw Data Files

This directory contains the original and curated data files used by VectorShop.

## Files

- `amazon.csv`: Primary dataset of Amazon products with reviews (1,465 products)
- `amazon_with_images.csv`: Enhanced dataset with validated image URLs (1,293 products with working image links)

## Dataset Structure

The dataset contains the following columns:

| Column | Description | Notes |
|--------|-------------|-------|
| product_id | Unique identifier for each product | Used as primary key |
| product_name | Full product name/title | |
| category | Product category hierarchy | Pipe-delimited hierarchical structure |
| discounted_price | Sale price in Indian Rupees (₹) | Converted to USD in processing |
| actual_price | Original price in Indian Rupees (₹) | |
| discount_percentage | Percentage discount | |
| rating | Product rating (0-5 scale) | Most products rated 4.0-4.3 |
| rating_count | Number of ratings received | |
| about_product | Product description | Often contains feature lists |
| user_id | ID of reviewer | |
| user_name | Name of reviewer | |
| review_id | Unique identifier for review | |
| review_title | Review title/headline | |
| review_content | Full review text | |
| product_link | URL to product page | |
| image_url | URL to product image | Validated and fixed in amazon_with_images.csv |

## Data Source

The data was sourced from Kaggle and includes product information, reviews, and pricing details from Amazon India. The original dataset has been enhanced with validated image URLs to ensure reliable image processing.

## Data Cleaning Notes

- Image URLs were validated and fixed where broken
- Price information was converted from Indian Rupees to USD using an exchange rate of 83 INR = 1 USD
- Category hierarchies were normalized for consistent formatting

## Usage

These files should not be modified directly. The processing scripts in the `vectorshop` package read from these files and generate the processed data in the appropriate directories.