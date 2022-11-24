from distutils.core import setup

setup(
    name='ProjectAssessment',
    version='0.2.8',
    packages=['ProjectAssessment',],
    python_requires='>3.7.0',
    license='MIT',
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'tqdm',
        'numba>=0.54.0',
        'scipy>=1.4.0',
    ],
    author='Ben Smith',
    author_email='bosmith@unomaha.edu',
    classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.7',
    ],
    keywords = ['Assessment', 'Projects', 'Statistics', 'Education', 'Bootstrap'],
    url = 'https://github.com/tazzben/project-based-assessment',
    download_url = 'https://github.com/tazzben/project-based-assessment/archive/v0.2.8.tar.gz',
    description = 'Package to compute the Project-Based Assessment estimates of student and rubric proficiency.',
)
