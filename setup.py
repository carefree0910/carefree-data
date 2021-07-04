import numpy
from Cython.Build import cythonize
from setuptools import setup, find_packages, Extension

VERSION = "0.2.5"

DESCRIPTION = "Data processing module implemented with numpy"
with open("README.md") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="carefree-data",
    version=VERSION,
    packages=find_packages(exclude=("tests",)),
    install_requires=[
        "optbinning",
        "datatable==0.11.1",
        "carefree-toolkit>=0.2.0",
        "dill",
        "future",
        "psutil",
        "pillow",
        "cython>=0.29.12",
        "numpy>=1.19.2",
        "scipy>=1.2.1",
        "scikit-learn>=0.20.3",
        "matplotlib>=3.0.3",
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
