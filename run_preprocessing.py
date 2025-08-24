#!/usr/bin/env python3
"""
Simple script to run the laptop data preprocessing pipeline.
This script demonstrates the processing of Amazon laptop reviews enriched dataset.
"""

import sys
import os
from data_preprocessing import LaptopDataPreprocessor
from data_explorer import LaptopDataExplorer
import pandas as pd

def main():
    """
    Main function to run the preprocessing pipeline and demonstrate results.
    """
    print("=" * 60)
    print("AMAZON LAPTOP REVIEWS ENRICHED DATASET PREPROCESSING PIPELINE")
    print("=" * 60)
    print("This script will:")
    print("1. Load the Amazon laptop reviews enriched dataset from Hugging Face")
    print("2. Clean and preprocess the data")
    print("3. Extract specifications from product details")
    print("4. Add derived features")
    print("5. Save the processed dataset")
    print("6. Generate analysis and visualizations")
    print("=" * 60)
    
    try:
        # Step 1: Initialize preprocessor
        print("\n1. Initializing data preprocessor...")
        preprocessor = LaptopDataPreprocessor()
        
        # Step 2: Run separated preprocessing pipeline for better data structure
        print("\n2. Running separated preprocessing pipeline...")
        df_laptop, df_rating = preprocessor.preprocess_separated_pipeline()
        
        print(f"✓ Preprocessing completed!")
        print(f"  - Laptop data: {len(df_laptop)} records")
        print(f"  - Rating data: {len(df_rating)} records")
        
        # Use laptop data as primary for analysis
        processed_data = df_laptop
        
        # Step 3: Display summary
        print("\n3. Generating data summary...")
        summary = preprocessor.get_separated_data_summary()
        
        print("\n" + "="*50)
        print("PREPROCESSING SUMMARY")
        print("="*50)
        print(f"Laptop Data:")
        print(f"  Total Products: {summary['laptop_data']['total_products']}")
        print(f"  Total Features: {summary['laptop_data']['total_features']}")
        print(f"  Number of Brands: {summary['laptop_data']['brands_count']}")
        if summary['laptop_data']['price_range_myr']['min'] is not None:
            print(f"  Price Range (MYR): RM {summary['laptop_data']['price_range_myr']['min']:.2f} - RM {summary['laptop_data']['price_range_myr']['max']:.2f}")
            print(f"  Average Price (MYR): RM {summary['laptop_data']['price_range_myr']['mean']:.2f}")
        if summary['laptop_data']['average_rating'] is not None:
            print(f"  Average Rating: {summary['laptop_data']['average_rating']:.2f}/5.0")
        
        print(f"\nRating Data:")
        print(f"  Total Reviews: {summary['rating_data']['total_reviews']}")
        print(f"  Total Features: {summary['rating_data']['total_features']}")
        print(f"  Unique Users: {summary['rating_data']['unique_users']}")
        print(f"  Unique Products: {summary['rating_data']['unique_products']}")
        if summary['rating_data']['rating_stats']['mean'] is not None:
            print(f"  Average Rating: {summary['rating_data']['rating_stats']['mean']:.2f}/5.0")
        
        # Step 4: Show sample of processed data
        print("\n4. Sample of processed laptop data:")
        print("-" * 50)
        sample_columns = ['brand', 'title_y', 'average_rating', 'price_myr']
        available_columns = [col for col in sample_columns if col in processed_data.columns]
        if available_columns:
            sample_data = processed_data[available_columns].head(5)
            for _, row in sample_data.iterrows():
                brand = row.get('brand', 'Unknown')
                title = row.get('title_y', 'No title')[:50] + "..." if len(str(row.get('title_y', ''))) > 50 else row.get('title_y', 'No title')
                rating = f"{row.get('average_rating', 'N/A')}/5" if pd.notna(row.get('average_rating')) else 'N/A'
                price = f"RM {row.get('price_myr', 'N/A'):.2f}" if pd.notna(row.get('price_myr')) else 'N/A'
                print(f"{brand}: {title} | Rating: {rating} | Price: {price}")
        
        # Step 5: Show feature columns
        print("\n5. Available laptop feature columns:")
        laptop_cols = list(processed_data.columns)
        print(f"  Total features: {len(laptop_cols)}")
        
        # Group columns by type
        basic_cols = [col for col in laptop_cols if any(x in col for x in ['asin', 'title', 'brand', 'os', 'color', 'store'])]
        price_cols = [col for col in laptop_cols if 'price' in col]
        rating_cols = [col for col in laptop_cols if 'rating' in col]
        encoded_cols = [col for col in laptop_cols if 'encoded' in col]
        clean_cols = [col for col in laptop_cols if 'clean' in col]
        
        print(f"  Basic info: {len(basic_cols)} features")
        print(f"  Pricing: {len(price_cols)} features")
        print(f"  Ratings: {len(rating_cols)} features")
        print(f"  Encoded: {len(encoded_cols)} features")
        print(f"  Cleaned: {len(clean_cols)} features")
        
        print("\n  Sample columns:")
        for col in laptop_cols[:10]:
            print(f"    - {col}")
        if len(laptop_cols) > 10:
            print(f"    ... and {len(laptop_cols)-10} more")
        
        # Step 6: Run data exploration
        print("\n6. Running data exploration...")
        explorer = LaptopDataExplorer()
        
        # Generate basic analysis
        price_analysis = explorer.analyze_price_distribution()
        brand_analysis = explorer.analyze_brands()
        rating_analysis = explorer.analyze_ratings()
        
        print("\n" + "="*50)
        print("QUICK ANALYSIS RESULTS")
        print("="*50)
        
        if 'error' not in price_analysis and price_analysis.get('price_categories'):
            print("Price Categories:")
            for category, count in price_analysis['price_categories'].items():
                print(f"  {category}: {count} laptops")
        
        if 'error' not in brand_analysis:
            print("\nTop 5 Brands:")
            for i, (brand, count) in enumerate(list(brand_analysis['top_brands'].items())[:5], 1):
                print(f"  {i}. {brand}: {count} reviews")
        
        if 'error' not in rating_analysis:
            print(f"\nRating Statistics:")
            stats = rating_analysis['rating_statistics']
            print(f"  Average Rating: {stats['mean']:.2f}/5.0")
            print(f"  Rating Range: {stats['min']:.1f} - {stats['max']:.1f}")
        
        # Step 7: Generate visualizations and report
        print("\n7. Generating visualizations and analysis report...")
        explorer.create_visualizations()
        explorer.generate_report()
        
        print("\n" + "="*60)
        print("✓ PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nOutput files created:")
        print(f"  - Processed laptop data: data/processed_laptop_data.csv")
        print(f"  - Visualizations: data/visualizations/")
        print(f"  - Analysis report: data/analysis_report.txt")
        print("\nKey Features:")
        print(f"  - {summary['laptop_data']['total_features']} laptop features including derived ones")
        print(f"  - {summary['laptop_data']['total_products']} clean laptop products")
        print(f"  - {summary['rating_data']['total_reviews']} user reviews")
        if summary['laptop_data']['price_range_myr']['min'] is not None:
            print(f"  - Price range: RM {summary['laptop_data']['price_range_myr']['min']:.2f} - RM {summary['laptop_data']['price_range_myr']['max']:.2f}")
        print(f"  - {summary['laptop_data']['brands_count']} different brands")
        if summary['laptop_data']['average_rating'] is not None:
            print(f"  - Average rating: {summary['laptop_data']['average_rating']:.2f}/5.0")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {e}")
        print("Please check the error message above and ensure the dataset is accessible.")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 All done! You can now use the processed data for your recommendation system.")
    else:
        print("\n💥 Preprocessing failed. Please check the error messages above.")
        sys.exit(1)
