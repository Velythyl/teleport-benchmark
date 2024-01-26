from setuptools import find_packages
from setuptools import setup

setup(
    name="teleported_benchmark",
    version="0.0.0",
    description=("2D Localization, Planning, and Kidnapping benchmarks"),
    author="Charlie Gauthier",
    author_email="charlie.gauthier@mila.quebec",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Velythyl/teleport-benchmark",
    license="GPLv3",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "pillow",
    ],
    keywords="Probabilistic sampling benchmark localization teleportation kidnapped robot problem",
)