# Data Caching System

## Overview

The laptop recommender system now includes a data caching system that eliminates the need to reprocess data every time the web application starts. This significantly improves startup time and reduces computational overhead.

## How It Works

### Before (Without Caching)
1. Web app starts
2. Data preprocessing runs (2-5 minutes)
3. Web app becomes available
4. Every restart requires full reprocessing

### After (With Caching)
1. **First run**: Data preprocessing runs and saves to cache files
2. **Subsequent runs**: Web app loads preprocessed data from cache (seconds)
3. **Cache management**: Automatic cache validation and refresh

## Cache Files

The system creates the following cache files in `data/cache/`:

- `laptop_data.parquet` - Processed laptop data (efficient binary format)
- `rating_data.parquet` - Processed rating data (efficient binary format)
- `cache_metadata.json` - Cache metadata (creation time, record counts, etc.)

## Usage

### 1. Preprocess Data Once (Recommended)

```bash
# Preprocess and cache data (run this once)
python preprocess_data.py

# Force reprocessing even if cache exists
python preprocess_data.py --force

# Check cache status
python preprocess_data.py --status

# Clear cache
python preprocess_data.py --clear
```

### 2. Start Web Application

```bash
# Normal startup (will use cached data if available)
python app.py
```

## Cache Management

### Automatic Cache Validation

The system automatically checks cache validity based on:

- **File existence**: All required cache files must exist
- **Age**: Cache is considered stale after 24 hours by default
- **Data integrity**: Cache files must be readable and contain expected data

### Manual Cache Management

```bash
# Check if cache exists and is fresh
python preprocess_data.py --status

# Force refresh cache (useful after data updates)
python preprocess_data.py --force

# Clear cache completely
python preprocess_data.py --clear
```

## Benefits

### Performance Improvements

- **Startup time**: Reduced from 2-5 minutes to 10-30 seconds
- **Memory usage**: More efficient data loading
- **CPU usage**: No repeated preprocessing
- **Storage**: Parquet format is more efficient than CSV

### Development Benefits

- **Faster iteration**: Quick restarts during development
- **Consistent data**: Same preprocessed data across runs
- **Offline capability**: Works without internet after initial preprocessing

## Configuration

### Cache Settings

You can modify cache behavior in `data_preprocessing.py`:

```python
# Cache directory
cache_dir = "data/cache"

# Maximum cache age (hours)
max_age_hours = 24

# Force reprocessing
force_reprocess = False
```

### Cache Invalidation

The cache is automatically invalidated when:

1. Cache files are missing or corrupted
2. Cache is older than `max_age_hours`
3. `force_reprocess=True` is specified
4. Data structure changes (detected via metadata)

## Troubleshooting

### Common Issues

1. **Cache not found**: Run `python preprocess_data.py` first
2. **Stale cache**: Run `python preprocess_data.py --force`
3. **Corrupted cache**: Run `python preprocess_data.py --clear` then `python preprocess_data.py`
4. **Memory issues**: Ensure sufficient RAM for data loading

### Debug Information

Check cache status for detailed information:

```bash
python preprocess_data.py --status
```

This shows:
- Cache creation time
- Data record counts
- Cache age
- File integrity status

## File Structure

```
data/
├── cache/                          # Cache directory
│   ├── laptop_data.parquet        # Cached laptop data
│   ├── rating_data.parquet        # Cached rating data
│   └── cache_metadata.json        # Cache metadata
├── processed_laptop_data.csv      # Legacy CSV output
└── analysis_report.txt            # Analysis report
```

## Migration from Old System

If you're upgrading from the old system:

1. **First run**: The system will automatically create cache files
2. **No data loss**: Original data processing logic remains unchanged
3. **Backward compatibility**: Old CSV files are still created
4. **Gradual adoption**: Can disable caching by setting `force_reprocess=True`

## Best Practices

1. **Preprocess once**: Run `python preprocess_data.py` after any data changes
2. **Monitor cache age**: Check status regularly in production
3. **Backup cache**: Include `data/cache/` in your backup strategy
4. **Version control**: Don't commit cache files (add to `.gitignore`)

## Technical Details

### Parquet Format

- **Efficiency**: 50-80% smaller than CSV
- **Speed**: 2-5x faster loading
- **Type safety**: Preserves data types
- **Compression**: Built-in compression

### Memory Management

- **Lazy loading**: Data loaded only when needed
- **Efficient storage**: Parquet's columnar format
- **Garbage collection**: Automatic cleanup of temporary objects

### Error Handling

- **Graceful fallback**: Falls back to preprocessing if cache fails
- **Detailed logging**: Comprehensive error messages
- **Recovery**: Automatic cache regeneration on errors
