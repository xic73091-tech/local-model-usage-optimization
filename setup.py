"""Setup configuration for local-model-optimizer."""

from setuptools import setup, find_packages
from pathlib import Path

here = Path(__file__).parent
long_description = (here / "README.md").read_text(encoding="utf-8") if (here / "README.md").exists() else ""

# Core dependencies (mirrors requirements.txt)
install_requires = [
    "fastapi>=0.100.0",
    "uvicorn[standard]>=0.23.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "llama-cpp-python>=0.2.0",
    "psutil>=5.9.0",
    "pyyaml>=6.0",
    "python-dotenv>=1.0.0",
    "huggingface-hub>=0.16.0",
    "httpx>=0.24.0",
    "aiofiles>=23.0.0",
    "rich>=13.0.0",
    "click>=8.1.0",
]

extras_require = {
    "gpu-monitor": ["pynvml>=11.5.0"],
    "gpu-monitor-amd": ["pyrsmi>=0.1.0"],
    "torch-backend": ["torch>=2.0.0", "transformers>=4.30.0"],
    "dev": [
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
        "ruff>=0.1.0",
        "mypy>=1.5.0",
    ],
}

setup(
    name="local-model-optimizer",
    version="0.1.0",
    description="Smart local LLM inference optimization toolkit - hardware detection, intelligent scheduling, and performance monitoring",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Local Model Optimizer Team",
    python_requires=">=3.10",
    packages=find_packages(where=".", include=["src", "src.*"]),
    package_dir={"": "."},
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "lmo-server=src.api.server:main",
            "lmo-hwinfo=src.core.hardware_detector:main",
            "lmo-benchmark=src.benchmark.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
