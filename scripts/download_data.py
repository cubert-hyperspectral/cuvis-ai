#!/usr/bin/env python3
"""Thin re-export of the dataset CLI from cuvis_ai_core.

The actual implementation lives in cuvis_ai_core.data.public_datasets.
"""

from cuvis_ai_core.data.public_datasets import download_data_cli as main

if __name__ == "__main__":
    main()
