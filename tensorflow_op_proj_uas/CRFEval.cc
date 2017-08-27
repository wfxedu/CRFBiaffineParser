/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <stddef.h>
#include <string>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/mutex.h"
#include <stdio.h>
#include <math.h>
#include "Eisner.h"
#include "TreeCRFLib.h"
#include "aligned-mem-pool.h"
#include "HelperCRF.h"

using tensorflow::shape_inference::InferenceContext;
using tensorflow::DEVICE_CPU;
using tensorflow::DT_INT32;
using tensorflow::DT_STRING;
using tensorflow::OpKernel;
using tensorflow::OpKernelConstruction;
using tensorflow::OpKernelContext;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::errors::InvalidArgument;

REGISTER_OP("CostOut")
    .Input("arc_scores: float32")
    .Input("sen_gold: int32")
    .Input("sen_lens: int32")
    .Output("out_prob: float32")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      return tensorflow::Status::OK();
    });
    
using namespace CRFParser;

#define DOUBLE_LARGENEG_P1 (-10000000.0+1)




std::mutex g_locker_eval;
AlignedMemoryPool<8>* mem_pool_eval() {
 static AlignedMemoryPool<8>*  s_mem_pool = 0;
 if(s_mem_pool==0) 
		s_mem_pool = new AlignedMemoryPool<8>(512UL * (1UL << 20));
   return s_mem_pool;
}

std::mutex g_locker_decode;
AlignedMemoryPool<8>* mem_pool_decode() {
 static AlignedMemoryPool<8>*  s_mem_pool = 0;
 if(s_mem_pool==0) 
		s_mem_pool = new AlignedMemoryPool<8>(512UL * (1UL << 20));
   return s_mem_pool;
}

void dump_matrix(double* marginals, int dim1,int dim2,int real_len) 
{
for(int i1=0;i1<dim1;i1++) {
					     for(int i2=0;i2<dim2;i2++) {
					        int idx1 = get_index2(real_len, i2, i1, 0, 1);
					        printf("%f,",(float)marginals[idx1]);
					     }
					     printf("\n");
					  }
}

class crf_eval : public OpKernel {
 public:
  explicit crf_eval(OpKernelConstruction *context) : OpKernel(context) {}

  // Counts term frequencies.
  void Compute(OpKernelContext *context) override {
    std::lock_guard<std::mutex> lck (g_locker_eval);
    
    const Tensor& arc_tensor = context->input(0);
    TensorShape arc_shape = arc_tensor.shape();
    auto arc_v = arc_tensor.flat<float>();
    
    const Tensor& sen_gold = context->input(1);
    auto gold_array = sen_gold.flat<int>();
    const TensorShape& gold_shape = sen_gold.shape();
    const Tensor& sen_len = context->input(2);
    auto len_array = sen_len.flat<int>();
        
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, arc_shape,&output_tensor));
    
    auto out_v = output_tensor->flat<float>();
    
    int batch_size = arc_shape.dim_size(0);
    int dim1 = arc_shape.dim_size(1);
    int dim2 = arc_shape.dim_size(2);
    for(int i=0;i<batch_size*dim1*dim2;i++)
       out_v(i) = 0;
    //LOG(WARNING) <<"batch_size="<< batch_size<<", dim1="<<dim1<<", dim2 = "<<dim2<<", label_size = "<<label_size;
    //LOG(WARNING) << arc_v ;
    //-----------------------------------------------------------------
   mem_pool_eval()->free();
  	long scores_size = dim1 * dim2 ;
	double* scores_label = (double*)mem_pool_eval()->allocate(scores_size * sizeof(double));
	double* marginals = (double*)mem_pool_eval()->allocate(dim1*dim1 * sizeof(double));

   double tt_z = 0,tt_p=0;
   for (int bi = 0;bi < batch_size;bi++) {
		for (int i = 0;i < scores_size;i++) {
			scores_label[i] = DOUBLE_LARGENEG_P1;
			marginals[i]=0;
		}


		int real_len = len_array(bi);
		if(real_len>dim1)
		   LOG(WARNING) <<"ERROR: len="<<real_len;
      //LOG(WARNING) <<"len="<<real_len;

		for (int m = 0;m < real_len;m++) {
			for (int h = 0;h < real_len;h++) {
			   int idx1= dim1*dim2*bi+dim2*m+h;
            float arc_src =  arc_v(idx1) ;
            int idx = get_index2(real_len, h, m, 0, 1);
				scores_label[idx] = arc_src;
			}
		}
		//-----------------------------
		double z = encodeMarginals(real_len, scores_label, marginals);
      //printf("{dim1=%d,dim2=%d}; ",(int)gold_shape.dim_size(1),(int)gold_shape.dim_size(2));
      tt_z+=z;
		//-------------------------------
		double sen_p =0;
		int ihit = 0;
		for (int m = 1;m < real_len;m++) {
		   int idx1= bi* gold_shape.dim_size(1) * gold_shape.dim_size(2) + m*gold_shape.dim_size(2);
			int g_h = gold_array(idx1 + 1);
			int g_r = gold_array(idx1 + 2);
			if(m==0)
			   g_h=0;

			//printf("h_%d,r_%d;",g_h,g_r);
			
			bool bhit = false;
			for (int h = 0;h < real_len;h++) {
  			   int idx = get_index2(real_len, h, m, 0, 1);
  			   double gs = 0;
				if (g_h == h ) {
					gs = -1;
					sen_p+=marginals[idx];ihit++;bhit=true;
					if(marginals[idx]>1.0001 || marginals[idx]<-0.0001) {
					   dump_matrix(marginals,dim1,dim2,real_len);
					  exit(-1);
					}
				}
				//int idxvv= dim1*dim2*bi+dim2*m+h;
			   int idxvv= dim1*dim2*bi+dim2*m+h;
			   out_v(idxvv) = (gs + marginals[idx]);
			}
			if(!bhit)
				  printf("h_%d,c_%d, sen_len %d ;",g_h,m, real_len);
		}
		if(ihit!=real_len-1)
		   printf("{bi=%d,hit=%d,len=%d}; ",bi,ihit,real_len);
		tt_p += sen_p/ihit;
		//printf("%f,",(float)(sen_p/real_len));
		//-------------------------------
	 }
	 LOG(WARNING) << "\tavr Z="<< tt_z/batch_size<<" avr logP="<<(tt_p/batch_size);
  }
};

REGISTER_KERNEL_BUILDER(Name("CostOut").Device(DEVICE_CPU), crf_eval);

//----------------------------------------------------------------------

REGISTER_OP("DecodeOut")
    .Input("arc_scores: float32")
    .Input("sen_lens: int32")
    .Output("out_pred: int32")
    .Output("out_rel: int32")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->UnknownShape());
      c->set_output(1, c->UnknownShape());
      return tensorflow::Status::OK();
    });

class crf_decode : public OpKernel {
 public:
  explicit crf_decode(OpKernelConstruction *context) : OpKernel(context) {}

  // Counts term frequencies.
  void Compute(OpKernelContext *context) override {
    std::lock_guard<std::mutex> lck (g_locker_decode);
    
    const Tensor& arc_tensor = context->input(0);
    TensorShape arc_shape = arc_tensor.shape();
    auto arc_v = arc_tensor.flat<float>();
    
    const Tensor& sen_len = context->input(1);
    auto len_array = sen_len.flat<int>();
        
    int batch_size = arc_shape.dim_size(0);
    int dim1 = arc_shape.dim_size(1);
    
    Tensor* opred_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({batch_size,dim1}),&opred_tensor)); 
    auto out_pred = opred_tensor->flat<int>();

    Tensor* orel_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({batch_size,dim1}),&orel_tensor)); 
    auto out_rel = orel_tensor->flat<int>();

    
    int dim2 = arc_shape.dim_size(2);
    //LOG(WARNING) <<"batch_size="<< batch_size<<", dim1="<<dim1<<", dim2 = "<<dim2<<", label_size = "<<label_size;
    //LOG(WARNING) << arc_v ;
    //-----------------------------------------------------------------
    mem_pool_decode()->free();
  	long scores_size = dim1 * dim2 ;
	double* scores_label = (double*)mem_pool_decode()->allocate(scores_size * sizeof(double));
   int* outsen = (int*)mem_pool_decode()->allocate(dim1 * 2 * sizeof(int));
   
   for (int bi = 0;bi < batch_size;bi++) {
		for (int i = 0;i < scores_size;i++)
			scores_label[i] = DOUBLE_LARGENEG_P1;

		int real_len = len_array(bi);
		//LOG(WARNING) <<"len="<<real_len;
		for (int m = 1;m < real_len;m++) {
			for (int h = 0;h < real_len;h++) {
			   int idx1= dim1*dim2*bi+dim2*m+h;
            float arc_src = arc_v(idx1);
            int idx = get_index2(real_len, h, m, 0, 1);
				scores_label[idx] = arc_src;
			}
		}
		//-----------------------------
		decodeProjective(real_len, scores_label, outsen);
		
		out_pred(bi*dim1+0) = 0;
		out_rel(bi*dim1+0) = 1;
		for (int m = 1;m < dim1;m++) {
			if (real_len > m) {
				out_pred(bi*dim1+m) = outsen[ m];
				out_rel(bi*dim1+m) = 0;
			} else {
				out_pred(bi*dim1+m) = 0;
				out_rel(bi*dim1+m) = 0;
			}
		}
		//-------------------------------
	 }
  }
};

REGISTER_KERNEL_BUILDER(Name("DecodeOut").Device(DEVICE_CPU), crf_decode);

//---------------------------------------------------------------------
REGISTER_OP("LabelOut")
    .Input("arc_scores: float32")
    .Input("rel_scores: float32")
    .Input("sen_lens: int32")
    .Output("out_pred: int32")
    .Output("out_rel: int32")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->UnknownShape());
      c->set_output(1, c->UnknownShape());
      return tensorflow::Status::OK();
    });

float log_clip(float v) {
   if(v<1e-10)
      v = 1e-10;
   return log(v);
}

class crf_decode_label : public OpKernel {
 public:
  explicit crf_decode_label(OpKernelConstruction *context) : OpKernel(context) {}

  // Counts term frequencies.
  void Compute(OpKernelContext *context) override {
    std::lock_guard<std::mutex> lck (g_locker_decode);
    
    const Tensor& arc_tensor = context->input(0);
    TensorShape arc_shape = arc_tensor.shape();
    auto arc_v = arc_tensor.flat<float>();
    
    const Tensor& rel_tensor = context->input(1);
    auto rel_v = rel_tensor.flat<float>();
    const TensorShape& rel_shape = rel_tensor.shape();
    const Tensor& sen_len = context->input(2);
    auto len_array = sen_len.flat<int>();
        
    int batch_size = rel_shape.dim_size(0);
    int dim1 = rel_shape.dim_size(1);
    
    Tensor* opred_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({batch_size,dim1}),&opred_tensor)); 
    auto out_pred = opred_tensor->flat<int>();

    Tensor* orel_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({batch_size,dim1}),&orel_tensor)); 
    auto out_rel = orel_tensor->flat<int>();

    
    int dim2 = rel_shape.dim_size(2);
    int label_size = rel_shape.dim_size(3);
    //LOG(WARNING) <<"batch_size="<< batch_size<<", dim1="<<dim1<<", dim2 = "<<dim2<<", label_size = "<<label_size;
    //LOG(WARNING) << arc_v ;
    //-----------------------------------------------------------------
    long label_offset = 2;
    int real_szlabel = label_size-label_offset;
    mem_pool_decode()->free();
  	long scores_size = dim1 * dim2 *label_size;
	double* scores_label = (double*)mem_pool_decode()->allocate(scores_size * sizeof(double));
   int* outsen = (int*)mem_pool_decode()->allocate(dim1 * 2 * sizeof(int));
   
   for (int bi = 0;bi < batch_size;bi++) {
		for (int i = 0;i < scores_size;i++)
			scores_label[i] = DOUBLE_LARGENEG_P1;

		int real_len = len_array(bi);
		//LOG(WARNING) <<"len="<<real_len;
		for (int m = 1;m < real_len;m++) {
			for (int h = 0;h < real_len;h++) {
			   int idx1= dim1*dim2*bi+dim2*m+h;
            float arc_src = arc_v(idx1);
				for (int l = 0;l < real_szlabel;l++) {
				   int idxvv= dim1*dim2*label_size*bi+dim2*label_size*m+label_size*h+l+label_offset;
					double rel_src = log_clip(rel_v(idxvv));                     
					int idx = get_index2(real_len, h, m, l, real_szlabel);
					scores_label[idx] = rel_src + arc_src;
				}
			}
		}
		//-----------------------------
		decodeProjectiveL(real_len, scores_label, real_szlabel, outsen);
		
		out_pred(bi*dim1+0) = 0; //for root
		out_rel(bi*dim1+0) = 1;
		for (int m = 1;m < dim1;m++) {
			if (real_len > m) {
				out_pred(bi*dim1+m) = outsen[2 * m];
				out_rel(bi*dim1+m) = outsen[2 * m + 1]+label_offset;
			} else {
				out_pred(bi*dim1+m) = 0;
				out_rel(bi*dim1+m) = 0;
			}
		}
		//-------------------------------
	 }
  }
};

REGISTER_KERNEL_BUILDER(Name("LabelOut").Device(DEVICE_CPU), crf_decode_label);

//----------------------------------------------------------------------



