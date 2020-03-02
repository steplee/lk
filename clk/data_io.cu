#include "helper_cuda.h"
#include "clk.h"

#include <torch/extension.h>

// We keep two global vars to cache consecuvtive calls and avoid allocations.
static ImagePyramid g_qpyr, g_rpyr;

static int tensors_to_texture(torch::Tensor& qrtensor, ImagePyramid* qpyr, ImagePyramid* rpyr);

RegistrationResult registerAffine_call_cuda(torch::Tensor& t) {
  ImagePyramid *qpyr=&g_qpyr, *rpyr=&g_rpyr;
  tensors_to_texture(t, qpyr,rpyr);

  return registerAffine_cuda(qpyr, rpyr);
}



ImagePyramid::~ImagePyramid() {
  for (int i=0; i<allocatedLevels; i++) {
    std::cout << "deallocating level " << i << std::endl;
    TORCH_CHECK(buf[i] != 0, "allocated level should not have null dev buffer");
    TORCH_CHECK(texs[i] != 0, "allocated level should not have null dev texture");
    nppiFree(buf[i]);
    cudaDestroyTextureObject(texs[i]);
    buf[i] = 0;
    texs[i] = 0;
  }
}

//const static float gaussian7[7] = { 1./64. , 6./64. , 15./64. , 20./64. , 15./64. , 6./64. , 1./64. };
const static float gaussian5[5] = { 1./16. , 4./16. , 6./16. , 4./16. , 1./16. };

inline static bool is_pow2(int a) { return (a & (a-1)) == 0; }


// the ImagePyramid arguments are outputs.
// Returns -1 if failed, or the level that we should use as first.
static int tensors_to_texture(torch::Tensor& qrtensor, ImagePyramid* qpyr, ImagePyramid* rpyr) {
  TORCH_CHECK(qrtensor.ndimension() == 4);
  TORCH_CHECK(qrtensor.size(0) == 2);
  TORCH_CHECK(qrtensor.size(1) == 1 or qrtensor.size(1) == 2); // RGB or gray+depth
  TORCH_CHECK(qrtensor.type().scalarType() == torch::ScalarType::Float);
  TORCH_CHECK(qrtensor.type().scalarType() == torch::ScalarType::Float);
  TORCH_CHECK(is_pow2(qrtensor.size(2))==true , "height must be a power of two");
  TORCH_CHECK(is_pow2(qrtensor.size(3))==true , "width must be a power of two");

  int baseW = qrtensor.size(3), baseH = qrtensor.size(2);
  int C = qrtensor.size(1);

  // Allocate.
  // TODO: Do not recreate if we are smaller, just shift pyramid over and return shift.
  if (qpyr->baseW != baseW) {
    int lvls = 4;
    qpyr->baseW = baseW; qpyr->baseH = baseH; qpyr->C = C; qpyr->lvls = lvls;
    rpyr->baseW = baseW; rpyr->baseH = baseH; rpyr->C = C; rpyr->lvls = lvls;
    qpyr->allocatedLevels = lvls;
    rpyr->allocatedLevels = lvls;

    for (int i=0; i<lvls; i++) {
      int width = baseW>>i, height = baseW>>i;
      std::cout << "Allocating " << i << " : " << width << " " << height << std::endl;
      //cudaMallocPitch(&qpyr->buf[i], &pitchBytes, width*sizeof(float), height);
      qpyr->buf[i] = nppiMalloc_32f_C1(width, height, &qpyr->pitch[i]);
      //std::cout << " got pitch " << qpyr->pitch[i] << std::endl;

      auto channelDesc = cudaCreateChannelDesc(32,C==3?32:0, 0, 0, cudaChannelFormatKindFloat);
      struct cudaResourceDesc resDesc;
      memset(&resDesc, 0, sizeof(resDesc));
      resDesc.resType = cudaResourceTypePitch2D;
      resDesc.res.pitch2D.devPtr = qpyr->buf[i];
      resDesc.res.pitch2D.desc = channelDesc;
      resDesc.res.pitch2D.width = width;
      resDesc.res.pitch2D.height = height;
      resDesc.res.pitch2D.pitchInBytes = qpyr->pitch[i];

      struct cudaTextureDesc texDesc;
      memset(&texDesc, 0, sizeof(texDesc));
      texDesc.addressMode[0]   = cudaAddressModeWrap;
      texDesc.addressMode[1]   = cudaAddressModeWrap;
      texDesc.filterMode       = cudaFilterModeLinear;
      texDesc.readMode         = cudaReadModeElementType;
      texDesc.normalizedCoords = 1;

      checkCudaErrors(cudaCreateTextureObject(&qpyr->texs[i], &resDesc, &texDesc, NULL));

      rpyr->buf[i] = nppiMalloc_32f_C1(width, height, &rpyr->pitch[i]);
      resDesc.res.pitch2D.devPtr = rpyr->buf[i];
      resDesc.res.pitch2D.pitchInBytes = rpyr->pitch[i];
      checkCudaErrors(cudaCreateTextureObject(&rpyr->texs[i], &resDesc, &texDesc, NULL));
    }
  }

  // Copy base level.
  checkCudaErrors(cudaMemcpy2D(qpyr->buf[0], qpyr->pitch[0],
      qrtensor.data<float>(), qpyr->baseW, qpyr->baseW, qpyr->baseW, cudaMemcpyHostToDevice));
  auto offset = (qrtensor.size(1) + qrtensor.size(2) + qrtensor.size(3)) * sizeof(float);
  checkCudaErrors(cudaMemcpy2D(rpyr->buf[0], qpyr->pitch[0],
      qrtensor.data<float>()+offset, qpyr->baseW, qpyr->baseW, qpyr->baseW, cudaMemcpyHostToDevice));

  // Make pyramid.
  int last_w = qpyr->baseW, last_h = qpyr->baseH;
  for (int i=1; i<qpyr->lvls; i++) {
    NppiSize srcSize { last_w , last_h };
    NppiPoint srcOffset { 0 , 0 };
    NppiSize roi { last_w , last_h };
    nppiFilterGaussPyramidLayerDownBorder_32f_C1R(
        qpyr->buf[i-1], qpyr->pitch[i-1], srcSize, srcOffset,
        qpyr->buf[i],   qpyr->pitch[i],   roi,
        2.0, 5, gaussian5, NPP_BORDER_CONSTANT);
    nppiFilterGaussPyramidLayerDownBorder_32f_C1R(
        rpyr->buf[i-1], rpyr->pitch[i-1], srcSize, srcOffset,
        rpyr->buf[i],   rpyr->pitch[i],   roi,
        2.0, 5, gaussian5, NPP_BORDER_CONSTANT);

    last_w >>= 1;
    last_h >>= 1;
    std::cout << "Making pyr " << i << " : " << last_w << " " << last_h << std::endl;
  }

  return 0;
}
