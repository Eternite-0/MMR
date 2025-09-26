#!/usr/bin/env python3
"""
Simple script to run the MMR Anomaly Detection GUI
"""

import sys
import os

def main():
    # Add the project root to the Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    
    # Import and run the GUI
    try:
        from gui_main import main as gui_main
        gui_main()
    except ImportError as e:
        print(f"Error importing GUI module: {e}")
        print("Make sure all dependencies are installed.")
        print("Try running: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error running GUI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()