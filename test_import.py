#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify module imports
"""

import os
import sys
import logging

def test_os_module():
    """Test os module functionality"""
    print("Testing os module:")
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_cache")
    print(f"Cache directory path: {cache_dir}")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        print(f"Created directory: {cache_dir}")
    else:
        print(f"Directory already exists: {cache_dir}")
    print("os module test completed successfully!")

if __name__ == "__main__":
    test_os_module() 