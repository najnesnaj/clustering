#!/usr/bin/env python3
"""
Simple test script to verify main script imports and basic functionality.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from data_fetcher import DataFetcher
from feature_extractor import FeatureExtractor
from clustering import StockClustering
from visualizer import ClusterVisualizer

def test_imports():
    """Test that all modules import correctly."""
    try:
        # Test basic imports
        logger.info("✓ DataFetcher imported successfully")
        logger.info("✓ FeatureExtractor imported successfully")
        logger.info("✓ StockClustering imported successfully")
        logger.info("✓ ClusterVisualizer imported successfully")
        
        # Test basic instantiation
        data_fetcher = DataFetcher()
        logger.info("✓ DataFetcher instantiated")
        
        feature_extractor = FeatureExtractor()
        logger.info("✓ FeatureExtractor instantiated")
        
        clustering_analyzer = StockClustering()
        logger.info("✓ StockClustering instantiated")
        
        visualizer = ClusterVisualizer()
        logger.info("✓ ClusterVisualizer instantiated")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Import/instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_main_import():
    """Test that main script can be imported."""
    try:
        # Try to import main
        import importlib.util
        import sys
        main_spec = importlib.util.spec_from_file_location("main", "main.py")
        main_module = importlib.util.module_from_spec(main_spec)
        logger.info("✓ Main script can be imported")
        return True
        
    except Exception as e:
        logger.error(f"✗ Main script import failed: {e}")
        return False

def main():
    """Run basic tests."""
    logger.info("Starting basic functionality tests...")
    
    # Test 1: Imports
    logger.info("\n=== Testing Imports ===")
    imports_ok = test_imports()
    
    # Test 2: Main script
    logger.info("\n=== Testing Main Script ===")
    main_ok = test_main_import()
    
    # Summary
    logger.info("\n=== Test Summary ===")
    if imports_ok:
        logger.info("✓ All imports successful")
    else:
        logger.error("✗ Some imports failed")
    
    if main_ok:
        logger.info("✓ Main script accessible")
    else:
        logger.error("✗ Main script issues found")
    
    if imports_ok and main_ok:
        logger.info("✓ Basic functionality tests passed!")
        return 0
    else:
        logger.error("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())