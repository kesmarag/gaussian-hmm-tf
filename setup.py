from setuptools import setup

setup(name='kesmarag-gaussian-hmm-tf',
      version='0.1.1',
      description='An implementation of the Gaussian Hidden Markov Model on top of TensorFlow',
      author='Costas Smaragdakis',
      author_email='kesmarag@gmail.com',
      url='https://github.com/kesmarag/gaussian-hmm-tf',
      packages=['kesmarag.ghmm'],
      package_dir={'kesmarag.ghmm': './'},
      install_requires=['tensorflow>=1.2.0',
                        'numpy>=1.12.1',
                        'scikit-learn>=0.18.1'], )
