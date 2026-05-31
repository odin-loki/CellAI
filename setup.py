"""
Cell AI — setup.py

Install with:
    pip install -e .
    (or) pip install .

Entry points:
    cell-ai         — unified CLI (all versions, all models, data pipelines)
"""

from setuptools import setup, find_packages

setup(
    name="cell-ai",
    version="1.0.0",
    description="Cell AI — cellular computation AI framework (v1/v2/v3)",
    author="Cell AI Project",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*", "docs*", "tools*"]),
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "tiktoken>=0.6.0",
        # CuPy: install cupy-cuda12x / cupy-cuda11x matching your CUDA version.
        # It is listed as optional because the CPU ThreadPoolExecutor fallback
        # works without it.  Add to extras_require["gpu"] if preferred.
        "transformers>=4.35.0",
        "datasets>=2.16.0",
        "python-dotenv>=1.0.0",
        "scipy>=1.11.0",
        "sympy>=1.12",
        "networkx>=3.1",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "datasketch>=1.6.0",
    ],
    extras_require={
        "gpu": ["cupy-cuda12x"],   # GPU-parallel partitions via CUDA streams
        "dev": ["pytest>=7.4.0", "pytest-asyncio"],
    },
    entry_points={
        "console_scripts": [
            "cell-ai=scripts.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
