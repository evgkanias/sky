import setuptools
import os

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fr:
    requirements = fr.read().splitlines()

setuptools.setup(
    name="skylight",
    version="v1.1",
    author="Evripidis Gkanias",
    maintainer="Evripidis Gkanias",
    author_email="ev.gkanias@ed.ac.uk",
    maintainer_email="ev.gkanias@ed.ac.uk",
    description="A package providing radiance, polarisation, and transmittance information from the daylight.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://pypi.org/project/skylight/",
    license="GPLv3+",
    project_urls={
        "Bug Tracker": "https://github.com/evgkanias/sky/issues",
        "Source": "https://github.com/evgkanias/sky"
    },
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
        "Operating System :: Microsoft :: Windows :: Windows 11",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Physics"
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    package_data={'': [
        "README.md",
        os.path.join('data', 'standard-parameters.yaml')
    ]},
    install_requires=requirements,
    python_requires=">=3.9",
)
