#ifdef WIN32
#define TREECRFLIB_API _declspec(dllexport)
#else
#define TREECRFLIB_API 
#endif


extern "C" TREECRFLIB_API void init_buffer(int size);


extern "C" TREECRFLIB_API float construct_grad(
	float * inA_array, long inA_strides[], long inA_shapes[],
	float * inL_array, long inL_strides[], long inL_shapes[],
	float * out_array, long out_strides[], long out_shapes[],
	int * gold_array, long gold_strides[], long gold_shapes[], 
	int * len_array, long len_strides[]);

extern "C" TREECRFLIB_API void decode(
	float * inA_array, long inA_strides[], long inA_shapes[],
	float * inL_array, long inL_strides[], long inL_shapes[],
	int * out_array, long out_strides[], long out_shapes[], long sen_rlen);
