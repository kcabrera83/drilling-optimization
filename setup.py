from setuptools import setup, find_packages

setup(
    name="drilling-optimization",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "flask>=3.0",
        "flask-cors>=4.0",
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
    ],
    author="Ing. Kelvin Cabrera",
    description="Sistema ML para optimizacion de parametros de perforacion petrolera",
)
