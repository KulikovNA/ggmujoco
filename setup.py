from setuptools import setup, find_packages
import os

setup(
    name="ggmujoco",
    version="1.0.0",
    description="Simulation of grasp fractured objects.",
    packages=find_packages(include=["ggmujoco", "ggmujoco.*"]),
    python_requires=">=3.8",
    include_package_data=True,  
)