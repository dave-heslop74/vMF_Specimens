import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vMF_Specimens",
    version="1.0.1",
    author="Dave Heslop",
    author_email="dave.heslop74@gmail.com",
    description="Fisher specimen uncertainty package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dave-heslop74",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
