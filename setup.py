from distutils.core import setup

setup(
    name='uncertainty',
    version='0.0.1',
    description='Classifier for detecting linguistic uncertainty in English.',
    long_description="",
    url='http://github.com/meyersbs/uncertainty',
    author='Benjamin S. Meyers',
    author_email='bsm9339@rit.edu',
    license='MIT',
    keywords=['nlp', 'natural language', 'natural language processing', 'uncertainty', 'classification'],
    packages=[
        'uncertainty',
        'uncertainty.corpora',
        'uncertainty.scripts',
    ],
    download_url='https://github.com/meyersbs/uncertainty/archive/v0.0.1.tar.gz',
    requires=['xml2json', 'scikit-learn', 'numpy'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Science/Research',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux'
    ]
)
