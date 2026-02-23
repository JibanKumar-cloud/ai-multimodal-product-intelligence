from setuptools import setup, find_packages

setup(
    name="wayfair-catalog-ai",
    version="0.1.0",
    description="Multimodal Product Attribute Extraction using Fine-tuned VLMs",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/YOUR_USERNAME/wayfair-catalog-ai",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "peft>=0.7.0",
        "bitsandbytes>=0.41.0",
        "pandas>=2.0.0",
        "Pillow>=10.0.0",
        "pyyaml>=6.0",
        "loguru>=0.7.0",
        "scikit-learn>=1.3.0",
        "openai>=1.6.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "ruff", "pytest-cov"],
        "demo": ["streamlit", "gradio"],
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
