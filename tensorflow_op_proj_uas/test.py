import tensorflow as tf
import numpy as np
crf_eval_module = tf.load_op_library('/home/wfx/parsing/tensorflow_op_proj_uas/CRFEval.so')

def crf_cost(arc_probs,rel_probs, targets, sen_len):
   out1 = crf_eval_module.cost_out(arc_probs,rel_probs,targets,sen_len)
   return tf.stop_gradient(out1)

def crf_decode(arc_probs,rel_probs, sen_len):
   arc_r,rel_r = crf_eval_module.decode_out(arc_probs,rel_probs,sen_len)
   return tf.stop_gradient(arc_r),tf.stop_gradient(rel_r)
    
if True:
   rnidx = 1
   x1 = np.random.uniform(-10,0, size=(2,3,3)).astype(np.float32)
   x2 = np.random.uniform(-10,0, size=(2,3,3,7)).astype(np.float32)
   len1 = np.random.randint(1,6,size = [2]).astype(np.int32)
   len1[0]=3
   len1[1]=3
   gold_conll = np.random.randint(0,3,size = [2,3,5]).astype(np.int32)

   tf_x1 = tf.get_variable('Weights1_%d' % rnidx, [2, 3,3], initializer=tf.constant_initializer(x1))
   tf_x2 = tf.get_variable('Weights2_%d' % rnidx, [2, 3,3,7], initializer=tf.constant_initializer(x2))
   tf_len1 = tf.get_variable('len1_%d'% rnidx, [2],dtype=tf.int32, initializer=tf.constant_initializer(len1))
   tf_gold_conll = tf.get_variable('gold_conll_%d' % rnidx, [2, 3,5],dtype=tf.int32, initializer=tf.constant_initializer(gold_conll))

       #tf_w = tf.constant(w_prob)
   #tf_arc = tf.expand_dims(tf_x1,3)
   #tf_res = (tf_arc+tf_x2)
   out1 = crf_cost(tf_x1,tf_x2,tf_gold_conll,tf_len1) #- tf_res
   cost = tf.reduce_sum(tf.reshape(out1,[-1]))
   tf_pred,tf_rel=crf_decode(tf_x1,tf_x2,tf_len1)

   with tf.Session('') as sess:
     sess.run(tf.initialize_all_variables())
     
     ar,o1,pred,rel=sess.run([cost,out1,tf_pred,tf_rel])
     print("cost=",ar)
     print(o1)
     print(pred)
     print(rel)
      


