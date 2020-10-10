from setuptools import setup

setup(
   name='rlephant',
   version='1.0',
   description='Simple tool to efficiently read and write episodes of MDP\'s to files.',
   author='Axel Brunnbauer',
   author_email='axel.brunnbauer@gmx.at',
   packages=['rlephant'],
   install_requires=['numpy', 'h5py']
)