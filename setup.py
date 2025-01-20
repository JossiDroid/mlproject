from setuptools import setup,find_packages
from typing import List
def get_requirements(file_path:str)->List[str]:

    requirements= []
    file_obj = open(file_path,'r')
    for lib in file_obj.readlines():
        if(lib != '-e .'):
            requirements.append(lib.replace('\n',''))

    return requirements


setup(
    name='mlproject',
    version='0.0.1',
    author='Jossel',
    author_email='jjo935576@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)