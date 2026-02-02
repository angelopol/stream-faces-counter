from setuptools import setup, find_packages

setup(
    name="stream-count-faces",
    version="1.0.0",
    description="Library for real-time face counting in video streams using AWS Rekognition.",
    author="angelopol",
    url="https://github.com/angelopol/stream-faces-counter",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.8.0",
        "boto3>=1.28.0",
        "pyyaml>=6.0",
        "numpy>=1.24.0",
        "python-dotenv>=1.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
