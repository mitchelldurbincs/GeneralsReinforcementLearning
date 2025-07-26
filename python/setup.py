from setuptools import setup, find_packages

setup(
    name="generals-pb",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "grpcio>=1.48.0",
        "protobuf>=3.20.0",
    ],
    python_requires=">=3.7",
    description="Python gRPC client for Generals Reinforcement Learning",
    author="Mitchell Durbin",
    license="MIT",
)
