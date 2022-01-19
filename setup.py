# Copyright 2017-2018 Lars Pastewka, Andreas Greiner
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Adopted from: https://github.com/pybind/python_example

from setuptools import find_packages, setup, Extension
from setuptools.command.build_ext import build_ext
import os
import sys
import setuptools

__version__ = '0.0.1'

#update
def get_eigen_include(eigen_version='3.4.0'):
    """Helper function to download and install eigen and return include path.
    """
    root = os.path.abspath(os.path.dirname(__file__))
    eigen_path = '{}/depend/eigen-{}'.format(root, eigen_version)
    if not os.path.exists(eigen_path):
        os.makedirs(eigen_path, exist_ok=True)
        #https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2
        #os.system('curl -L http://bitbucket.org/eigen/eigen/get/{}.tar.bz2 | tar -jx -C {} --strip-components 1'.format(eigen_version, eigen_path))
        # TODO this seems to be broken, ill copy manually
        #os.system('curl -L https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2 | tar'.format(eigen_version, eigen_path))
    return(eigen_path)


def get_pybind11_include(pybind11_version='2.2.3'):
    """Helper function to download and install pybind and return include path.
    """
    root = os.path.abspath(os.path.dirname(__file__))
    pybind11_path = '{}/depend/pybind11-{}'.format(root, pybind11_version)
    if not os.path.exists(pybind11_path):
        os.makedirs(pybind11_path, exist_ok=True)
        os.system('curl -L https://github.com/pybind/pybind11/archive/v{}.tar.gz | tar -zx -C {} --strip-components 1'.format(pybind11_version, pybind11_path))
    return('{}/include'.format(pybind11_path))


ext_modules = [
    Extension(
        '_lbkernels',
        ['c/_lbkernels.cpp'],
        include_dirs=[
            # Path to pybind11 and Eigen headers
            get_eigen_include(),
            get_pybind11_include()
        ],
        language='c++'
    ),
]


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.
    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)


setup(
    name='PyLB',
    version=__version__,
    author='Lars Pastewka',
    author_email='lars.pastewka@imtek.uni-freiburg.de',
    url='https://github.com/pastewka/LBWithPython',
    description='Lattice Boltzmann with Python',
    long_description='',
    packages = find_packages(),
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
    test_suite='tests'
)
