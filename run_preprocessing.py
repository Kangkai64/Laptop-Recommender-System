#!/usr/bin/env python3
"""
Simple script to run the laptop data preprocessing pipeline.
This script demonstrates the currency conversion from INR to MYR.
"""

import sys
import os
from data_preprocessing import LaptopDataPreprocessor
from data_explorer import LaptopDataExplorer

def main():
    """
    Main function to run the preprocessing pipeline and demonstrate results.
    """
    print("=" * 60)
    print("LAPTOP DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    print("This script will:")
    print("1. Load the original laptop dataset")
    print("2. Clean and preprocess the data")
    print("3. Convert prices from INR to MYR")
    print("4. Add derived features")
    print("5. Save the processed dataset")
    print("6. Generate analysis and visualizations")
    print("=" * 60)
    
    try:
        # Step 1: Initialize preprocessor
        print("\n1. Initializing data preprocessor...")
        preprocessor = LaptopDataPreprocessor()
        
        # Step 2: Run complete preprocessing pipeline
        print("\n2. Running preprocessing pipeline...")
        processed_data = preprocessor.preprocess_pipeline()
        
        print(f"✓ Preprocessing completed! Processed {len(processed_data)} records")
        
        # Step 3: Display summary
        print("\n3. Generating data summary...")
        summary = preprocessor.get_data_summary()
        
        print("\n" + "="*50)
        print("PREPROCESSING SUMMARY")
        print("="*50)
        print(f"Total Records: {summary['total_records']}")
        print(f"Total Features: {summary['total_features']}")
        print(f"Exchange Rate (INR to MYR): {summary['exchange_rate_used']:.4f}")
        print(f"Price Range (MYR): RM {summary['price_range_myr']['min']:.2f} - RM {summary['price_range_myr']['max']:.2f}")
        print(f"Average Price (MYR): RM {summary['price_range_myr']['mean']:.2f}")
        print(f"Number of Brands: {summary['brands_count']}")
        
        # Step 4: Show sample of converted prices
        print("\n4. Sample of price conversion (INR → MYR):")
        print("-" * 50)
        sample_data = processed_data[['brand', 'model', 'latest_price', 'latest_price_myr']].head(10)
        for _, row in sample_data.iterrows():
            print(f"{row['brand']} {row['model']}: ₹{row['latest_price']:,.0f} → RM {row['latest_price_myr']:,.2f}")
        
        # Step 5: Show feature columns
        print("\n5. Available feature columns:")
        feature_cols = preprocessor.export_feature_columns()
        for category, columns in feature_cols.items():
            print(f"  {category}: {len(columns)} features")
            if len(columns) <= 5:  # Show all if 5 or fewer
                for col in columns:
                    print(f"    - {col}")
            else:  # Show first 3 and last 2
                for col in columns[:3]:
                    print(f"    - {col}")
                print(f"    ... ({len(columns)-5} more) ...")
                for col in columns[-2:]:
                    print(f"    - {col}")
        
        # Step 6: Run data exploration
        print("\n6. Running data exploration...")
        explorer = LaptopDataExplorer()
        
        # Generate basic analysis
        price_analysis = explorer.analyze_price_distribution()
        brand_analysis = explorer.analyze_brands()
        
        print("\n" + "="*50)
        print("QUICK ANALYSIS RESULTS")
        print("="*50)
        print("Price Categories:")
        for category, count in price_analysis['price_categories'].items():
            print(f"  {category}: {count} laptops")
        
        print("\nTop 5 Brands:")
        for i, (brand, count) in enumerate(list(brand_analysis['top_brands'].items())[:5], 1):
            print(f"  {i}. {brand}: {count} models")
        
        # Step 7: Generate visualizations and report
        print("\n7. Generating visualizations and analysis report...")
        explorer.create_visualizations()
        explorer.generate_report()
        
        print("\n" + "="*60)
        print("✓ PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nOutput files created:")
        print(f"  - Processed data: data/processed_laptop_data.csv")
        print(f"  - Visualizations: data/visualizations/")
        print(f"  - Analysis report: data/analysis_report.txt")
        print("\nKey Features:")
        print(f"  - Currency converted from INR to MYR (rate: {summary['exchange_rate_used']:.4f})")
        print(f"  - {summary['total_features']} features including derived ones")
        print(f"  - {summary['total_records']} clean laptop records")
        print(f"  - Price range: RM {summary['price_range_myr']['min']:.2f} - RM {summary['price_range_myr']['max']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {e}")
        print("Please check the error message above and ensure the data file exists.")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 All done! You can now use the processed data for your recommendation system.")
    else:
        print("\n💥 Preprocessing failed. Please check the error messages above.")
        sys.exit(1)
