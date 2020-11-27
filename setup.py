from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'gcsfs',
    'pandas',
    'scikit-learn',
    'google-cloud-storage',
    'pygeohash',
    'category_encoders',
    'mlflow',
    'joblib',
    'numpy',
    'psutil',
    'pygeohash',
    'termcolor',
    'xgboost',
    'memoized-property',
    'scipy',
    'category_encoders',
    'six',
    'pip',
    'setuptools',
    'wheel',
      'pandas',
      'pytest',
      'coverage',
      'flake8',
      'black',
      'yapf',
      'python-gitlab',
      'twine',
      'gensim',
      'tensorflow',
      'nltk',
      'sklearn']


setup(name='Fake_news',
      version="1.0",
      install_requires=REQUIRED_PACKAGES,
      description="Project Description",
      packages=find_packages(),
      test_suite='tests',
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      scripts=['scripts/Fake_news-run'],
      zip_safe=False)
