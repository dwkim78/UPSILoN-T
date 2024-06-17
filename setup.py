from setuptools import find_packages, setup


setup(
    name='upsilont',
    version='0.1.0',
    description='UPSILoN-T',
    long_description='',
    platforms=['any'],
    packages=find_packages(),
    include_package_data=True,
    url='https://github.com/dwkim78/UPSILoN-T',
    license='Apache v2.0',
    author='Dae-Won Kim',
    author_email='dwk@etri.re.kr',
    install_requires=['scikit-learn==1.5.0', 'numpy>=1.17',
                      'scipy>=1.3', 'pandas>=0.25'],
    keywords=['Machine Learning', 'Periodic Variable Stars', 'Time Series'],
    classifiers=[
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ]
)
