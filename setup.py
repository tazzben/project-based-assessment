from distutils.core import setup

setup(
    name='ProjectAssessment',
    version='0.0.1',
    packages=['ProjectAssessment',],
    license='MIT',
    install_requires=[
        'numpy',
        'pandas',
        'progress',
        'scipy',
        'prettytable',
    ],
    author='Ben Smith',
    author_email='bosmith@unomaha.edu',
    classifiers=[
    'Development Status :: 3 - Alpha', 
    'Intended Audience :: Science/Research', 
    'License :: OSI Approved :: MIT License',  
    'Programming Language :: Python :: 3.9',
    ],
    keywords = ['Assessment', 'Projects', 'Statistics', 'Education', 'Bootstrap'],
    url = 'https://github.com/tazzben/project-based-assessment',
    download_url = 'https://github.com/tazzben/project-based-assessment/archive/v0.0.4.tar.gz',  
    description = 'Package to compute the Project-Based Assessment estimates of student and rubric proficiency.',
)
