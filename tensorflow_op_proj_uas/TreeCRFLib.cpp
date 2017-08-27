// TreeCRFLib.cpp : Defines the exported functions for the DLL application.
//

#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
// Windows Header Files:
#include <stdio.h>
#include <math.h>
#include "Eisner.h"
#include "TreeCRFLib.h"
#include "aligned-mem-pool.h"

static AlignedMemoryPool<8>*  mem_pool = 0;
#define DOUBLE_LARGENEG_P1 (-10000000.0+1)
using namespace CRFParser;

TREECRFLIB_API void init_buffer(int size)
{
   if(mem_pool==0) {
		mem_pool = new AlignedMemoryPool<8>(320UL * (1UL << 20));
		long scores_size = 100;
		double* scores_label = (double*)mem_pool->allocate(scores_size * sizeof(double));
		scores_label[1] = 100;
   }
}

TREECRFLIB_API float construct_grad(
	float * inA_array, long inA_strides[], long inA_shapes[],
	float * inL_array, long inL_strides[], long inL_shapes[],
	float * out_array, long out_strides[], long out_shapes[],
	int * gold_array, long gold_strides[], long gold_shapes[],
	int * len_array, long len_strides[])
{
	mem_pool->free();
	//int sen_size = inA_shapes[1];
	int label_size = inL_shapes[3];
		

	long scores_size = inA_shapes[1] * inA_shapes[1] *label_size;
	double* scores_label = (double*)mem_pool->allocate(scores_size * sizeof(double));
	//--------------------------------------
	int ia_S0 = inA_strides[0] / sizeof(float);
	int ia_S1 = inA_strides[1] / sizeof(float);
	int ia_S2 = inA_strides[2] / sizeof(float);

	int il_S0 = inL_strides[0] / sizeof(float);
	int il_S1 = inL_strides[1] / sizeof(float);
	int il_S2 = inL_strides[2] / sizeof(float);
	int il_S3 = inL_strides[3] / sizeof(float);

	int ig_S0 = gold_strides[0] / sizeof(int);
	int ig_S1 = gold_strides[1] / sizeof(int);
	int ig_S2 = gold_strides[2] / sizeof(int);

	int o_S0 = out_strides[0] / sizeof(float);
	int o_S1 = out_strides[1] / sizeof(float);
	int o_S2 = out_strides[2] / sizeof(float);
	int o_S3 = out_strides[3] / sizeof(float);
	
	int len_s0 = len_strides[0]/sizeof(int);
	int ii = 0;
	float res_v = 0;
	for (int bi = 0;bi < inA_shapes[0];bi++) {
		for (int i = 0;i < scores_size;i++)
			scores_label[i] = DOUBLE_LARGENEG_P1;

		int real_len = len_array[bi*len_s0];

		for (int m = 0;m < real_len;m++) {
			for (int h = 0;h < real_len;h++) {
				double arc_src = inA_array[bi*ia_S0 + m*ia_S1 + h*ia_S2];
				if(arc_src<1e-10) arc_src = 1e-10;
				arc_src = log(arc_src);
				for (int l = 0;l < label_size;l++) {
					double rel_src = inL_array[bi*il_S0 + m*il_S1 + h*il_S2 + (l)*il_S3];
               if(rel_src<1e-10) rel_src = 1e-10;   
               rel_src = log(rel_src);                      
					int idx = get_index2(real_len, h, m, l, label_size);
					scores_label[idx] = rel_src + arc_src;
				}
			}
		}
		//-------------------------------
		double* marginals = (double*)mem_pool->allocate(real_len*real_len*label_size * sizeof(double));
		double* marginals_pure = (double*)mem_pool->allocate(real_len*real_len * sizeof(double));
		double z = LencodeMarginals(real_len, scores_label, label_size, marginals, marginals_pure);
      //printf("{len=%d,lbl=%d, z=%f}; ",real_len,label_size,z);
		//-------------------------------
		for (int m = 0;m < real_len;m++) {
			int g_h = gold_array[bi*ig_S0 + m*ig_S1 + 1 * ig_S2];
			int g_r = gold_array[bi*ig_S0 + m*ig_S1 + 2 * ig_S2];
			for (int h = 0;h < real_len;h++) {
				for (int l = 0;l < label_size;l++) {
					int idx = get_index2(real_len, h, m, l, label_size);

					double gs = 0;
					if (g_h == h && g_r == l) {
						gs = -1;
					}

					out_array[bi*o_S0 + m*o_S1 + h*o_S2 + (l)*o_S3] = (gs + marginals[idx]);
				}
			}
		}
		//-------------------------------
      for (int m = 0;m < real_len;m++) {
			for (int h = 0;h < real_len;h++) {
			   double arc_src = inA_array[bi*ia_S0 + m*ia_S1 + h*ia_S2];
			   if(arc_src<1e-10) arc_src = 1e-10;
				arc_src = log(arc_src);
				for (int l = 0;l < label_size;l++) {
				   double rel_src = inL_array[bi*il_S0 + m*il_S1 + h*il_S2 + (l)*il_S3];
				   if(rel_src<1e-10) rel_src = 1e-10;   
               rel_src = log(rel_src); 
					int idx = get_index2(real_len, h, m, l, label_size);
					
					res_v+= (arc_src+rel_src)*out_array[bi*o_S0 + m*o_S1 + h*o_S2 + (l)*o_S3];
				}
			}
		}
	}
	//--------------------------------------
	FILE* fout = fopen("./del_test.log","wb");
   for (int bi = 0;bi < inA_shapes[0];bi++) {
      int real_len = len_array[bi*len_s0];
   	for (int m = 0;m < real_len;m++) {
   	   for (int h = 0;h < real_len;h++) {
   	     for (int l = 0;l < label_size;l++) {
   	       double rel_src = out_array[bi*o_S0 + m*o_S1 + h*o_S2 + (l)*o_S3];
   	       fprintf(fout,"%f,\n",rel_src);
   	     }
   	   }
   	}
   }
	fclose(fout);
	//--------------------------------------
	printf("res_v = %e\n",res_v);
	return res_v;
}



TREECRFLIB_API void decode(
	float * inA_array, long inA_strides[], long inA_shapes[],
	float * inL_array, long inL_strides[], long inL_shapes[],
	int * out_array, long out_strides[], long out_shapes[],  long sen_rlen)
{
	mem_pool->free();
	//int sen_size = inA_shapes[0];
	int label_size = inL_shapes[2];
   //printf("(sen %d, label %d, strides %d", sen_rlen,label_size,inA_strides[1]);

	long scores_size = sen_rlen*sen_rlen*label_size;
	double* scores_label = (double*)mem_pool->allocate(scores_size * sizeof(double));
	int* outsen = (int*)mem_pool->allocate(sen_rlen * 2 * sizeof(int));
	//--------------------------------------
	int ia_S0 = inA_strides[0] / sizeof(float);
	int ia_S1 = inA_strides[1] / sizeof(float);
	//int ia_S2 = inA_strides[2] / sizeof(float);

	int il_S0 = inL_strides[0] / sizeof(float);
	int il_S1 = inL_strides[1] / sizeof(float);
	int il_S2 = inL_strides[2] / sizeof(float);
	//int il_S3 = inL_strides[3] / sizeof(float);


	int o_S0 = out_strides[0] / sizeof(int);
	int o_S1 = out_strides[1] / sizeof(int);
	//int o_S2 = out_strides[2] / sizeof(float);
	//--------------------------------------
	int ii = 0;
	for (int i = 0;i < scores_size;i++)
		scores_label[i] = DOUBLE_LARGENEG_P1;

	for (int m = 0;m < sen_rlen;m++) {
		for (int h = 0;h < sen_rlen;h++) {
			double arc_src = inA_array[m*ia_S0 + h*ia_S1];
         if(arc_src<1e-10) arc_src = 1e-10;
				arc_src = log(arc_src);
			for (int l = 0;l < label_size;l++) {
				double rel_src = inL_array[m*il_S0 + h*il_S1 + l*il_S2];
				if(rel_src<1e-10) rel_src = 1e-10;   
               rel_src = log(rel_src); 
				int idx = get_index2(sen_rlen, h, m, l, label_size);
				scores_label[idx] = rel_src + arc_src;
			}
		}
	}
	//-------------------------------
        //printf("\n%d,%d,%d,%d\n",sizeof(int),sizeof(float),sizeof(long long),sizeof(long));
        //out_array[0] = 100;
	decodeProjectiveL(sen_rlen, scores_label, label_size, outsen);
	for (int m = 0;m < inA_shapes[0];m++) {
		if (sen_rlen > m) {
			out_array[m*o_S0 + 0 * o_S1] = outsen[2 * m];
			out_array[m*o_S0 + 1 * o_S1] = outsen[2 * m + 1];
		} else {
			out_array[m*o_S0 + 0 * o_S1] = 0;
			out_array[m*o_S0 + 1 * o_S1] = 0;
		}
	}

}
