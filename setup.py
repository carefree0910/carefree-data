import numpy
from Cython.Build import cythonize
from setuptools import setup, find_packages, Extension

VERSION = "0.2.6"

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
        "carefree-toolkit>=0.2.9",
        "dill",
        "future",
        "psutil",
        "pillow",
        "cython>=0.29.28",
        "numpy>=1.22.3",
        "scipy>=1.8.0",
        "scikit-learn>=1.0.2",
        "matplotlib>=3.5.1",
    ],
    ext_modules=cythonize(
        Extension(
            "cfdata.misc.c.cython_utils",
            sources=["cfdata/misc/c/cython_utils.pyx"],
            language="c",
            include_dirs=[numpy.get_include(), "cfdata/misc/c"],
            library_dirs=[],
            libraries=[],
            extra_compile_args=[],
            extra_link_args=[],
        )
    ),
    package_data={"cfdata.misc.c": ["cython_utils.pyx"]},
    author="carefree0910",
    author_email="syameimaru_kurumi@pku.edu.cn",
    url="https://github.com/carefree0910/carefree-data",
    download_url=f"https://github.com/carefree0910/carefree-data/archive/v{VERSION}.tar.gz",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    keywords="python numpy data-science",
)
