'''
The setup.py file is an essential part of packing and distributing python projects.
It is used by setuptools to define the configuration of your project such as its metadata,
dependencies, and etc.
'''

from setuptools import find_packages, setup
from typing import List

def get_requirements() -> List[str]:
    """
    This function will return list of requirements
    """
    requirement_list:List[str] = []  # Empty List of strings
    try:
        with open('requirements.txt','r') as file:
            # Read lines from the file
            lines = file.readlines()
            # Process each line
            for line in lines:
                requirement = line.strip() # To remove any empty space
                # Ignore empty line and -e .
                if requirement and requirement!='-e .':
                    requirement_list.append(requirement)
    except FileNotFoundError:
        print("Requirements.txt file not found.")

    return requirement_list

setup(
    name="NetworkSecurity",
    version="0.0.1",
    author="Nidhish Kaul",
    packages=find_packages(),
    install_requires=get_requirements()
)
