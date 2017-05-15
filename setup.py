from setuptools import setup

setup(
        name='uncertainty',
        version='0.1.0',
        packages=['uncertainty', 'uncertainty.data', 'uncertainty.features'],
        package_data={
            'uncertainty': ['models/*.p', 'vectorizers/*.p']
        },
        install_requires=[
            'numpy==1.12.1',
            'scipy==0.19.0',
            'scikit-learn==0.18.1',
            'nltk==3.2.2'
        ],
        license='The MIT License (MIT) Copyright (c) 2017 Benjamin Meyers',
        description='Python implementation of a classifier for linguistic '
                    'uncertainty, based on the work by Vincze et al.',
        author='Benjamin Meyers',
        author_email='bsm9339@rit.edu',
        url='https://github.com/meyersbs/uncertainty',
        test_suite='uncertainty.tests',
        classifiers=[]
    )
