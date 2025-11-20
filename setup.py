from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ran-optimizer",
    version="0.1.0",
    author="RADCOM Team",
    description="RAN optimization and recommendation system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests*", "legacy*", "explore*"]),
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "ipython>=8.0.0",
            "jupyter>=1.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "ran-optimize=ran_optimizer.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Telecommunications Industry",
        "Programming Language :: Python :: 3.11",
    ],
)
