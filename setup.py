from setuptools import find_packages, setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(
    name='upsilont',
    version='0.1.0',
    description='UPSILoN-T',
    long_description=readme(),
    platforms=['any'],
    packages=find_packages(),
    include_package_data=True,
    url='',
    license='Apache v2.0',
    author='Dae-Won Kim',
    author_email='dwk@etri.re.kr',
    install_requires=['scikit-learn>=0.22', 'numpy>=1.17',
                      'scipy>=1.3', 'pandas>=0.25'],
    keywords=['Machine Learning', 'Periodic Variable Stars', 'Time Series'],
    classifiers=[
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ]
)
