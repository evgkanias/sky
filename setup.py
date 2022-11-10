import setuptools
import os

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="sky",
    version="v1.0-beta",
    author="Evripidis Gkanias",
    maintainer="Evripidis Gkanias",
    author_email="ev.gkanias@ed.ac.uk",
    maintainer_email="ev.gkanias@ed.ac.uk",
    description="A package providing radiance, polarisation, and transmittance information from the daylight.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/evgkanias/sky",
    project_urls={
        "Bug Tracker": "https://github.com/evgkanias/sky/issues"
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.9",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Licence :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
        "Operating System :: Microsoft :: Windows :: Windows 11"
    ],
    packages=["sky", "skygui"],
    package_dir={"sky": "src/sky",
                 "skygui": "src/skygui"},
    package_data={'sky': [os.path.join('data', 'PragueSkyModelDatasetGroundInfra.dat'),
                          os.path.join('data', 'standard-parameters.yaml')],
                  'skygui': [os.path.join('data', 'icon.png')]},
    python_requires=">=3.9",
)
