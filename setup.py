from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# read the contents of your README file
with open("README.txt", encoding="utf-8") as f:
    readme = f.read()

setup(
    name="leurn-demo",
    version="0.1",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            # list your console scripts here
            "demo = leurn.demo:main"
        ],
    },
    author="Caglar Aytekin",
    author_email="cagosmail@gmail.com",
    description="LEURN Learning Explainable Univariate Rules with Neural Networks",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/CaglarAytekin/LEURN_DEMO",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: Attribution-NonCommercial-NoDerivatives 4.0 International",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
