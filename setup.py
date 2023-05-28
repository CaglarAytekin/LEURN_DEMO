from setuptools import find_packages, setup

# read the contents of your requirements file
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# read the contents of your README file
with open("README.md", encoding="utf-8") as f:
    readme = f.read()

# read the contents of your LICENSE file
with open("LICENSE-CC-BY-NC-ND-4.0.md", encoding="utf-8") as f:
    license = f.read()

# get the package version from __init__.py
packages = find_packages()
with open(f"{packages[0]}/__init__.py", encoding="utf-8") as f:
    exec(f.readline())

setup(
    name="leurn-demo",
    version=globals().get("__version__", "0.1"),
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            # list your console scripts here
            "leurn-demo = leurn.demo:main",
            # just for debugging
            "leurn-hello = leurn.demo:helloworld",
        ],
    },
    author="Caglar Aytekin",
    author_email="cagosmail@gmail.com",
    description="LEURN Learning Explainable Univariate Rules with Neural Networks",
    long_description=readme,
    long_description_content_type="text/markdown",
    license=license,
    url="https://github.com/CaglarAytekin/LEURN_DEMO",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: Attribution-NonCommercial-NoDerivatives 4.0 International",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    # so housing.xlsx is included
    include_package_data=True,
)
