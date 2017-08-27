#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np
crf_eval_module = tf.load_op_library('/home/wfx/parsing/tensorflow_op_proj/CRFEval.so')

def crf_cost(arc_probs,rel_probs, inputs,targets, sen_len):
   tf_w,tf_ashape,tf_rshape  = crf_eval_module.cost_out(arc_probs,rel_probs,inputs,targets,sen_len)
   return tf.stop_gradient(tf_w),tf.stop_gradient(tf_ashape),tf.stop_gradient(tf_rshape)

def crf_decode(arc_probs,rel_probs, sen_len):
   arc_r,rel_r = crf_eval_module.decode_out(arc_probs,rel_probs,sen_len)
   return tf.stop_gradient(arc_r),tf.stop_gradient(rel_r)

def crf_decode_uas(arc_probs, sen_len):
   arc_r = crf_eval_module.unlabel_out(arc_probs,sen_len)
   return tf.stop_gradient(arc_r)





   
