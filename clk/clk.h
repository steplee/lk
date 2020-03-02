#pragma once
#include <npp.h>

/*
 *
 * I had to split up the code in a non-intuitive way to factor out torch
 * code to its own .cu because it is SLOW to compile it's include
 * there is a   registerAffine_cuda() that does the work (clk_impl.cu)
 * and a simple registerAffine_call_cuda() that directs the call.
 *  ---> Actually it looks like setuptools doesn't cache unchanged objects, so ....
 *
 */

struct RegistrationResult {
  RegistrationResult();
};


struct ImagePyramid {

  int baseW=0, baseH=0;
  int C = 0;

  int lvls=0;
  int allocatedLevels=0;
  Npp32f* buf[12] = {0}; // We never should have larger than 12 levels.
  int pitch[12] = {0};
  cudaTextureObject_t texs[12] = {0};

  bool valid=false;

  ~ImagePyramid();
};

RegistrationResult registerAffine_cuda(ImagePyramid* qpyr, ImagePyramid* rpyr);
