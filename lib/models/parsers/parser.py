#!/usr/bin/env python
# -*- coding: UTF-8 -*-
 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import crflibUnlabel

from vocab import Vocab
from lib.models.parsers.base_parser import BaseParser

#***************************************************************
class Parser(BaseParser):
  """"""
  
  #=============================================================
  def __call__(self, dataset, moving_params=None):
    """"""
    
    vocabs = dataset.vocabs
    inputs = dataset.inputs
    targets = dataset.targets
    
    reuse = (moving_params is not None)
    self.tokens_to_keep3D = tf.expand_dims(tf.to_float(tf.greater(inputs[:,:,0], vocabs[0].ROOT)), 2)
    self.sequence_lengths = tf.reshape(tf.reduce_sum(self.tokens_to_keep3D, [1, 2]), [-1,1])
    self.n_tokens = tf.reduce_sum(self.sequence_lengths)
    self.moving_params = moving_params
    
    word_inputs, pret_inputs = vocabs[0].embedding_lookup(inputs[:,:,0], inputs[:,:,1], moving_params=self.moving_params)
    tag_inputs = vocabs[1].embedding_lookup(inputs[:,:,2], moving_params=self.moving_params)
    if self.add_to_pretrained:
      word_inputs += pret_inputs
    if self.word_l2_reg > 0:
      unk_mask = tf.expand_dims(tf.to_float(tf.greater(inputs[:,:,1], vocabs[0].UNK)),2)
      word_loss = self.word_l2_reg*tf.nn.l2_loss((word_inputs - pret_inputs) * unk_mask)
    embed_inputs = self.embed_concat(word_inputs, tag_inputs)

    loctokens_to_keep3D = tf.expand_dims(tf.to_float(tf.greater_equal(inputs[:,:,0], vocabs[0].ROOT)), 2)
    locsequence_lengths = tf.reshape(tf.reduce_sum(loctokens_to_keep3D, [1, 2]), [-1,1]) 
    batch_size = tf.reduce_sum(tf.to_float(tf.greater_equal(locsequence_lengths, 0)))
    
    top_recur = embed_inputs
    for i in xrange(self.n_recur):
      with tf.variable_scope('RNN%d' % i, reuse=reuse):
        top_recur, _ = self.RNN(top_recur)
    
    with tf.variable_scope('MLP', reuse=reuse):
      dep_mlp, head_mlp = self.MLP(top_recur, self.class_mlp_size+self.attn_mlp_size, n_splits=2)
      dep_arc_mlp, dep_rel_mlp = dep_mlp[:,:,:self.attn_mlp_size], dep_mlp[:,:,self.attn_mlp_size:]
      head_arc_mlp, head_rel_mlp = head_mlp[:,:,:self.attn_mlp_size], head_mlp[:,:,self.attn_mlp_size:]
    
    with tf.variable_scope('Arcs', reuse=reuse):
      arc_logits = self.bilinear_classifier(dep_arc_mlp, head_arc_mlp)
      arc_output = self.output(arc_logits, targets[:,:,1])
      predictions_real,_ =  crflibUnlabel.crf_decode( arc_logits , tf.to_int32(locsequence_lengths) )
      if moving_params is None:
        predictions = targets[:,:,1]
      else:
        predictions = predictions_real
        
    with tf.variable_scope('Rels', reuse=reuse):
      rel_logits, rel_logits_cond = self.conditional_bilinear_classifier(dep_rel_mlp, head_rel_mlp, len(vocabs[2]), predictions)
      rel_output = self.output(rel_logits, targets[:,:,2])
      rel_output['probabilities'] = self.conditional_probabilities(rel_logits_cond)
      rel_logits4D = self.conditional_logits4D(rel_logits_cond)
    
    output = {}
    output['probabilities'] = [arc_output['probabilities'],rel_output['probabilities']]
    
    #arc_pred,rel_pred = crflib.crf_decode( arc_logits ,rel_logits4D, tf.to_int32(locsequence_lengths) )
    arc_pred, rel_pred = predictions_real,rel_output['predictions_train']
    arc_predictions1D = tf.to_int32(tf.reshape(arc_pred,[-1]))     
    arc_correct1D = tf.to_float(tf.equal(arc_predictions1D, arc_output['targets1D'] ))
    arc_n_correct = tf.reduce_sum(arc_correct1D * arc_output['tokens_to_keep1D'] )
    arc_accuracy = arc_n_correct / arc_output['n_tokens']
    
    output['predictions'] = [arc_pred,rel_pred]  #notuse
    output['correct'] = arc_correct1D * arc_output['tokens_to_keep1D']
    output['tokens'] = arc_output['tokens_to_keep1D']
    output['n_correct'] = arc_n_correct
    output['n_tokens'] = self.n_tokens
    output['accuracy'] = arc_accuracy
    
    arc_pred_test,rel_pred_test = crflibUnlabel.crf_decode_label( arc_logits ,rel_output['probabilities'], tf.to_int32(locsequence_lengths) )
    output['predictions_test'] = [arc_pred_test,rel_pred_test] 
    output['probabilities_test'] = [arc_logits ,rel_output['probabilities']]
    
    tf_w = crflibUnlabel.crf_cost( arc_logits ,targets, tf.to_int32(locsequence_lengths ) )
    cost_crf = tf.reduce_sum(tf.reshape(arc_logits*tf_w,[-1]))/ self.n_tokens
  
    output['loss'] = cost_crf +  rel_output['loss_train']
    if self.word_l2_reg > 0:
      output['loss'] += word_loss
    
    output['embed'] = embed_inputs
    output['recur'] = top_recur
    output['dep_arc'] = dep_arc_mlp
    output['head_dep'] = head_arc_mlp
    output['dep_rel'] = dep_rel_mlp
    output['head_rel'] = head_rel_mlp
    output['arc_logits'] = arc_logits
    output['rel_logits'] = rel_logits
    return output
  
  #=============================================================
  def prob_argmax(self, parse_probs, rel_probs, tokens_to_keep):
    """"""
    
    parse_preds = self.parse_argmax(parse_probs, tokens_to_keep)
    rel_probs = rel_probs[np.arange(len(parse_preds)), parse_preds]
    rel_preds = self.rel_argmax(rel_probs, tokens_to_keep)
    return parse_preds, rel_preds
