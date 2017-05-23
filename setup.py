from setuptools import setup

setup(
        name='uncertainty',
        version='0.1.0',
        packages=[
            'uncertainty', 'uncertainty.data', 'uncertainty.lib',
            'uncertainty.lib.nlp'
        ],
        package_data={
            'uncertainty': ['models/*.p', 'vectorizers/*.p'],
            'uncertainty.lib.nlp': ['verbs.txt']
        },
        install_requires=[
            'numpy==1.12.1',
            'scipy==0.19.0',
            'scikit-learn==0.18.1',
            'nltk==3.2.2'
        ],
        license='The MIT License (MIT) Copyright (c) 2017 Benjamin S. Meyers',
        description='Python implementation of a classifier for linguistic '
                    'uncertainty, based on the work by Vincze et al.',
        author='Benjamin S. Meyers',
        author_email='bsm9339@rit.edu',
        url='https://github.com/meyersbs/uncertainty',
        #download_url='https://github.com/meyersbs/uncertainty/uncertainty/dist/uncertainty-0.1.0.tar.gz',
        test_suite='uncertainty.tests',
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Natural Language :: English',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Topic :: Scientific/Engineering :: Information Analysis',
            'Topic :: Software Development :: Libraries :: Python Modules',
            'Topic :: Text Processing :: Linguistic'
        ],
        keywords=[
            'nlp', 'natural language', 'natural language processing',
            'uncertainty', 'linguistic uncertainty', 'classification'
        ]
    )
