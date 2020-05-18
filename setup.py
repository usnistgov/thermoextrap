
from setuptools import setup

setup(name='libExtrap',
      version='0.1',
      packages=['libextrap',],
      description="Library of functions useful for thermodynamic extrapolation" \
                  " and interpolation",
      long_description=open('README.md').read(),
      license=open('LICENSE').read(),
      install_requires=["numpy",
                        "scipy",
                        "sympy",
                       ],
     )
