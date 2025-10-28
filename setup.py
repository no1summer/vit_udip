from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vit-udip",
    version="1.0.0",
    author="no1summer",
    author_email="steveissummer@gmail.com",
    description="Vision Transformer for Unsupervised Deep Image Processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/no1summer/vit-udip",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "nibabel>=3.2.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-image>=0.19.0",
        "tqdm>=4.62.0",
        "tensorboard>=2.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)