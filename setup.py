from setuptools import setup
from setuptools import find_packages


VERSION = "0.2.9"

DESCRIPTION = "Data processing module implemented with numpy"
with open("README.md") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="carefree-data",
    version=VERSION,
    packages=find_packages(exclude=("tests",)),
    install_requires=[
        "optbinning",
        "datatable>=1.0.0",
        "carefree-toolkit>=0.2.11",
        "carefree-cython>=0.1.1",
        "scikit-learn>=1.0.2",
    ],
    author="carefree0910",
    author_email="syameimaru.saki@gmail.com",
    url="https://github.com/carefree0910/carefree-data",
    download_url=f"https://github.com/carefree0910/carefree-data/archive/v{VERSION}.tar.gz",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    keywords="python numpy data-science",
)
