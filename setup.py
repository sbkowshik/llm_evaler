from setuptools import setup, find_packages

setup(
    name="llm_evaler",
    version="0.1.0",
    description="A tool for aligning LLM evaluations with human preferences",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "datasets>=3.5.0",
        "transformers>=4.51.0",
        "openai>=1.72.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.13.0",
        "numpy>=1.23.0",
        "pandas>=2.0.0",
        "tqdm>=4.66.0",
        "pathlib>=1.0.1",
        "python-dotenv>=1.0.0",
        "argparse>=1.4.0",
        "pytest>=7.4.0",
        "ruff>=0.1.0",
        "streamlit>=1.38.0",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "llm-evaler=llm_evaler.app.app:main",
        ],
    },
) 