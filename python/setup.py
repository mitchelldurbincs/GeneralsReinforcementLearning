from setuptools import setup, find_packages

setup(
    name="generals-rl",
    version="0.1.0",
    packages=find_packages(include=["generals_agent", "generals_agent.*", "generals_pb", "generals_pb.*"]),
    package_dir={"": "."},
    install_requires=[
        "grpcio>=1.48.0",
        "protobuf>=3.20.0",
        "numpy>=1.24.0",
    ],
    python_requires=">=3.8",
    description="Python gRPC client and RL agents for Generals Reinforcement Learning",
    author="GeneralsRL Team",
    license="MIT",
)
