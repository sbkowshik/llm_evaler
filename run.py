#!/usr/bin/env python
"""
Run script for the LLM Evaluator application.
"""

import subprocess
import sys
import os
from pathlib import Path


def main():
    """
    Run the Streamlit application directly or via installed package.
    """
    try:
        # Check if we're in development mode
        if Path("llm_evaler/app/app.py").exists():
            # Run streamlit on the app module directly
            result = subprocess.run(
                [
                    "streamlit", 
                    "run", 
                    "llm_evaler/app/app.py"
                ],
                check=True
            )
            return result.returncode
            
        # If we're in installed mode, try the entry point
        else:
            # Try to import the module
            from llm_evaler.app import app
            
            # Run the app with streamlit
            result = subprocess.run(
                [
                    "streamlit", 
                    "run", 
                    os.path.abspath(app.__file__)
                ],
                check=True
            )
            return result.returncode

    except ImportError:
        print("Error: Could not import the llm_evaler package.")
        print("Please install the package with 'pip install -e .'")
        return 1
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to run the Streamlit application: {e}")
        return e.returncode
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 