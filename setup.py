from setuptools import setup

setup(
    name="rafft",
    version="2.0",
    description="FFT-based RNA folding",
    authors="Vaitea Opuu, Nono S. C. Merleau, Matteo Smerlak",
    license="MIT",
    packages=["rafft"],
    scripts=["bin/rafft", "bin/rafft_kin"],
    include_package_data=True)
