from setuptools import setup, find_packages

setup(
    name='isingmetro',
    version='0.1.0',
    packages = ['isingmetro'], 
    install_requires=['numpy', 'matplotlib', 'tqdm'],
    author='SuvamT0071',
    author_email='profsuvam@gmail.com',
    description='Ising Model Simulation using Metropolis-Hastings Algorithm',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/SuvamT0071/ising2D.git',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
