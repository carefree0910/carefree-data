import numpy
from Cython.Build import cythonize
from setuptools import setup, find_packages, Extension

setup(
    name="carefree-data",
    version="0.1.0",
    packages=find_packages(exclude=("tests",)),
    install_requires=[
        "carefree-toolkit",
        "dill", "future", "psutil", "pillow",
        "cython>=0.29.12", "numpy>=1.16.2", "scipy>=1.2.1",
        "scikit-learn>=0.20.3", "matplotlib>=3.0.3",
        "mkdocs", "mkdocs-material", "mkdocs-minify-plugin",
        "Pygments", "pymdown-extensions"
    ],
    ext_modules=cythonize(Extension(
        "cfdata.misc.c.cython_utils",
        sources=["cfdata/misc/c/cython_utils.pyx"],
        language="c",
        include_dirs=[numpy.get_include()],
        library_dirs=[],
        libraries=[],
        extra_compile_args=[],
        extra_link_args=[]
    )),
    author="carefree0910",
    author_email="syameimaru_kurumi@pku.edu.cn",
    description="Data processing module implemented with numpy",
    keywords="python numpy data-science"
)
