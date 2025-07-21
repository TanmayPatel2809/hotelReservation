from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()


setup(
    name = "hotelReservation",
    version = "1.0",
    author = "Tanmay",
    packages = find_packages(),
    install_requires = requirements,
)