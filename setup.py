import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='rlephant',
    version='1.0.0',
    description='A simple tool to efficiently read and write episodes of MDP\'s to files.',
    author='Axel Brunnbauer',
    author_email='axel.brunnbauer@gmx.at',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/axelbr/rlephant",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy', 'h5py']
)
