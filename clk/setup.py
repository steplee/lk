from setuptools import setup, Extension
from torch.utils.cpp_extension import CUDAExtension, BuildExtension,CppExtension

setup(name='clk',
      ext_modules=[CUDAExtension(
          'clk',
          [ 'clk.cpp','data_io.cu', 'clk_impl.cu' ],
          libraries = 'nppi nppc nppif nppisu'.split()
        )],
      cmdclass={'build_ext': BuildExtension})
