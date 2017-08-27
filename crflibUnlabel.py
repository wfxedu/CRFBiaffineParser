#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np
crf_eval_module = tf.load_op_library('/home/wfx/parsing/tensorflow_op_proj_uas/CRFEval.so')

def crf_cost(arc_probs, targets, sen_len):
   out1 = crf_eval_module.cost_out(arc_probs,targets,sen_len)
   return tf.stop_gradient(out1)

def crf_decode(arc_probs, sen_len):
   arc_r,rel_r = crf_eval_module.decode_out(arc_probs,sen_len)
   return tf.stop_gradient(arc_r),tf.stop_gradient(rel_r)


def crf_decode_label(arc_probs,rel_probs, sen_len):
   arc_r,rel_r = crf_eval_module.label_out(arc_probs,rel_probs,sen_len)
   return tf.stop_gradient(arc_r),tf.stop_gradient(rel_r)




   
