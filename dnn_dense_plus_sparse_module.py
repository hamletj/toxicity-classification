#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 16:16:19 2020

@author: ji.l
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class DnnModel():
    """
    Fully connected neural network with 2 parts in parallel
    """

    def __init__(self,
                 LEARNING_RATE=0.001,
                 BATCH_SIZE=32,
                 EVA_STEP=10,
                 SAVE_STEP=1000,
                 NUM_EPOCHS=3,
                 BETA=0.0000001,
                 KEEP_PROB = 0.7):
        self.LEARNING_RATE = LEARNING_RATE
        self.BATCH_SIZE = BATCH_SIZE
        self.EVA_STEP = EVA_STEP
        self.SAVE_STEP = SAVE_STEP
        self.NUM_EPOCHS = NUM_EPOCHS
        self.BETA = BETA
        self.KEEP_PROB = KEEP_PROB

    ###########################################
    # Build a graph                           #
    # Input --> Hidden Layer --> Output Layer #
    ###########################################

    def dense_graph(self, x, input_dim, DENSE_HIDDEN_DIM1, DENSE_HIDDEN_DIM2):
        """
        Structured data layer
        The output will be used to merge with text data layer output
        input:
            x (matrix) - structured data
            input_dim (scalar) - number of columns in x
        output:
            hidden2 (matrix) - dim==(, HIDDEN2_DIM_S)
        """

        w1 = tf.Variable(tf.random_uniform([input_dim, DENSE_HIDDEN_DIM1], minval=-1, maxval=1), name='weights1')
        b1 = tf.Variable(tf.zeros([DENSE_HIDDEN_DIM1]), name='bias1')
        hidden1 = tf.identity(tf.nn.sigmoid(tf.matmul(x, w1)+b1), name='hidden1')

        w2 = tf.Variable(tf.random_uniform([DENSE_HIDDEN_DIM1, DENSE_HIDDEN_DIM2], minval=-1, maxval=1), name='weights2')
        b2 = tf.Variable(tf.zeros([DENSE_HIDDEN_DIM2]), name='bias2')
        hidden2 = tf.identity(tf.nn.sigmoid(tf.matmul(hidden1, w2)+b2), name='hidden2')
        hidden2 = tf.nn.dropout(hidden2, self.KEEP_PROB)

        return hidden2
    
    def sparse_graph(self, input_indices, input_wd_ids, input_wd_wts, vocab_size, SPARSE_HIDDEN_DIM1, SPARSE_HIDDEN_DIM2):
        """
        Text data layer
        The output will be used to merge with structured data layer output
        input:
            input_indices - nonzero indices of a sparse matrix vecs
            input_wd_ids - word ids corresponding to nonzero indices
            input_wd_wts - word weights such as tfidf
            vocab_size - vocabulary size, e.g. from tfidf vectorizer
        output:
            hidden2 (matrix) - dim==(, HIDDEN2_DIM_T)
        """

        w1 = tf.Variable(tf.random_uniform([vocab_size, SPARSE_HIDDEN_DIM1], minval=-1, maxval=1), name='weights3')
        b1 = tf.Variable(tf.zeros([SPARSE_HIDDEN_DIM1]), name='bias3')

        # Create tf sparestensor from indices and values to store wd_ids. Dense shape is needed to tell the shape of the represented dense matrix.
        sparse_ids=tf.SparseTensor(indices=input_indices,
                                      values=input_wd_ids,
                                      dense_shape=[self.BATCH_SIZE, vocab_size])
        # Create tf sparestensor from indices and values to store wd_wts.
        sparse_wts=tf.SparseTensor(indices=input_indices,
                                      values=input_wd_wts,
                                      dense_shape=[self.BATCH_SIZE, vocab_size])

        hidden1=tf.identity(tf.nn.embedding_lookup_sparse(w1, sparse_ids, sparse_wts, combiner = "sum")+b1, name='hidden3')
        hidden1=tf.nn.dropout(hidden1, self.KEEP_PROB)

        w2 = tf.Variable(tf.random_uniform([SPARSE_HIDDEN_DIM1, SPARSE_HIDDEN_DIM2], minval=-1, maxval=1), name='weights4')
        b2 = tf.Variable(tf.zeros([SPARSE_HIDDEN_DIM2]), name='bias4')
        hidden2 = tf.identity(tf.matmul(hidden1, w2)+b2, name='hidden4')
        hidden2 = tf.nn.dropout(hidden2, self.KEEP_PROB)

        return hidden2
    
    def dense_sparse_concat_graph(self, layer1, layer2, num_cates, DENSE_HIDDEN_DIM2, SPARSE_HIDDEN_DIM2, CONCAT_HIDDEN_DIM):
        """
        Merge structured data layer and text data layer
        num_cates (scalar) - number of unique output categories == output dimension
        """

        x = tf.concat([layer1, layer2], 1)
        w1 = tf.Variable(tf.random_uniform([DENSE_HIDDEN_DIM2+SPARSE_HIDDEN_DIM2, CONCAT_HIDDEN_DIM], minval=-1, maxval=1), name='weights5')
        b1 = tf.Variable(tf.zeros([CONCAT_HIDDEN_DIM]), name='bias5')
        hidden1 = tf.identity(tf.matmul(x, w1)+b1, name='hidden5')
        w_out = tf.Variable(tf.random_uniform([CONCAT_HIDDEN_DIM, num_cates], minval=-1, maxval=1), name='concat_weights_out')
        b_out = tf.Variable(tf.zeros([num_cates]), name='concat_bias_out')
        logits = tf.identity(tf.matmul(hidden1, w_out)+b_out, name='logits')
        probs=tf.nn.softmax(logits, name='concat_probs')

        return w1, w_out, logits, probs

    def dnn_train(self,
                  inputX,
                  vecs,
                  targetY,
                  pathname_to_save,
                  DENSE_HIDDEN_DIM1,
                  DENSE_HIDDEN_DIM2,
                  SPARSE_HIDDEN_DIM1, 
                  SPARSE_HIDDEN_DIM2,
                  CONCAT_HIDDEN_DIM,
                  REGULARIZATION=False
                  ):
        """
        The training pipeline, from features to model
        input:
            inputX - input array (sample_size, input_dim)
            targetY - target array (sample_size, output_dim)
            pathname_to_save - model path and name
        """

        # Parameters
        input_dim = inputX.shape[1]
        output_dim = targetY.shape[1]
        #train_steps = int(inputX.shape[0]/self.BATCH_SIZE)+1
        train_steps = int(len(targetY)/self.BATCH_SIZE)+1 
        vocab_size = vecs.shape[1]

        # Build Graph
        tf.reset_default_graph()
        x = tf.placeholder(tf.float32, shape=[None, input_dim], name='inputs')
        y = tf.placeholder(tf.float32, shape=[None, output_dim], name='targets')
        input_indices = tf.placeholder(tf.int64, name='indices') # Indices of nonzero tfidf. Shape = [Unknown,2]
        input_wd_ids = tf.placeholder(tf.int64, name='wd_ids') # wd_ids are simply the second values of indices
        input_wd_wts = tf.placeholder(tf.float64, name='wd_wts') # a wd_wt is the tfidf value at an index
        
        dense_layer = self.dense_graph(x, input_dim, DENSE_HIDDEN_DIM1, DENSE_HIDDEN_DIM2)
        sparse_layer = self.sparse_graph(input_indices, 
                                         input_wd_ids, 
                                         input_wd_wts, 
                                         vocab_size, 
                                         SPARSE_HIDDEN_DIM1, 
                                         SPARSE_HIDDEN_DIM2)
        
        w1, w_out, logits, probs = self.dense_sparse_concat_graph(dense_layer, 
                                                                  sparse_layer, 
                                                                  output_dim, 
                                                                  DENSE_HIDDEN_DIM2, 
                                                                  SPARSE_HIDDEN_DIM2, 
                                                                  CONCAT_HIDDEN_DIM)

        # Loss function and optimizer
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
        regularizer = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w_out)
        if REGULARIZATION == True:
            loss = tf.reduce_mean(loss+self.BETA*regularizer, name='loss')
        else:
            loss = tf.reduce_mean(loss, name='loss')
        optimizer = tf.train.RMSPropOptimizer(self.LEARNING_RATE).minimize(loss)

        # Run Graph
        saver = tf.train.Saver(max_to_keep=3)
        sess = tf.Session()
        print('Graph started running...')
        print('There will be %d epochs. Each eopch will have %d steps.' %(self.NUM_EPOCHS, train_steps))
        sess.run(tf.global_variables_initializer()) # this could be very slow with large w and large output_dim
        for ep in range(self.NUM_EPOCHS):
            dense_batch_gen = self.get_dense_batch(inputX, targetY, self.BATCH_SIZE)
            sparse_batch_gen = self.get_sparse_batch(vecs, targetY, self.BATCH_SIZE, output_dim)
            total_loss=0.0
            for step in range(train_steps):
                #print((step+1) + train_steps*ep)
                dense_data_batch, tg_batch = next(dense_batch_gen)
                index_batch, wd_wt_batch, _ = next(sparse_batch_gen)
                wd_id_batch = index_batch[:,1]
                loss_batch, _ = sess.run([loss, optimizer],
                                         feed_dict={
                                         x: dense_data_batch,
                                         y: tg_batch,
                                         input_indices: index_batch,
                                         input_wd_ids: wd_id_batch,
                                         input_wd_wts: wd_wt_batch})
                total_loss += loss_batch
                # Evaluate Training Data
                if ((step+1) + train_steps*ep) % self.EVA_STEP == 0: 
                    print('Average loss at Epoch %d and Step %d is: %f' %(ep, step, total_loss/self.EVA_STEP))
                    total_loss=0.0
                if ((step+1) + train_steps*ep) % self.SAVE_STEP == 0: 
                    saver.save(sess, pathname_to_save, global_step=(step+1) + train_steps*(ep))
                    print("Model saved to: %s" % pathname_to_save)


    def dnn_eval(self,
                 inputX,
                 vecs,
                 targetY,
                 model_path,
                 model_name):
        """
        Restore model
        Run the input data and evaluate accuracy using given labels
        input:
            inputX (dense matrix)
            targetY (list) - known labels
            model_path - tensorflow model path
            model_name - tensorflow model name
        """

        dense_batch_gen = self.get_dense_batch(inputX, targetY, targetY.shape[0])
        dense_data_batch, tg_batch = next(dense_batch_gen)
        sparse_batch_gen = self.get_sparse_batch(vecs, targetY, targetY.shape[0], targetY.shape[1])
        index_batch, wd_wt_batch, _ = next(sparse_batch_gen)
        wd_id_batch = index_batch[:,1]

        # Load Model
        tf.reset_default_graph()
        sess = tf.Session()
        saver = tf.train.import_meta_graph(model_path+model_name)
        saver.restore(sess, tf.train.latest_checkpoint(model_path))

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name('inputs:0')
        y = graph.get_tensor_by_name('targets:0')
        input_indices = graph.get_tensor_by_name('indices:0')
        input_wd_ids = graph.get_tensor_by_name('wd_ids:0')
        input_wd_wts = graph.get_tensor_by_name('wd_wts:0')

        probs = graph.get_tensor_by_name('concat_probs:0')
        loss = graph.get_tensor_by_name('loss:0')

        # Run the model
        feed_dict={
                x: dense_data_batch,
                y: tg_batch,
                input_indices: index_batch,
                input_wd_ids: wd_id_batch,
                input_wd_wts: wd_wt_batch
                }
        test_probs, test_loss = sess.run([probs, loss], feed_dict)
        print("Test loss:", test_loss)
        return test_probs

    ########################################################
    ####-- Train Functions Only Using Sparse Feature  --####
    ########################################################
    def sparse_graph_sparseOnly(self, 
                     input_indices, 
                     input_wd_ids, 
                     input_wd_wts, 
                     vocab_size, 
                     SPARSE_HIDDEN_DIM1, 
                     num_cates):
        """
        Text data layer
        The output will be used to merge with structured data layer output
        input:
            input_indices - nonzero indices of a sparse matrix vecs
            input_wd_ids - word ids corresponding to nonzero indices
            input_wd_wts - word weights such as tfidf
            vocab_size - vocabulary size, e.g. from tfidf vectorizer
        output:
            logits (matrix) - dim==(, HIDDEN2_DIM_T)
        """

        w1 = tf.Variable(tf.random_uniform([vocab_size, SPARSE_HIDDEN_DIM1], minval=-1, maxval=1), name='sparse_weights1')
        b1 = tf.Variable(tf.zeros([SPARSE_HIDDEN_DIM1]), name='sparse_bias1')

        # Create tf sparestensor from indices and values to store wd_ids. Dense shape is needed to tell the shape of the represented dense matrix.
        sparse_ids=tf.SparseTensor(indices=input_indices,
                                      values=input_wd_ids,
                                      dense_shape=[self.BATCH_SIZE, vocab_size])
        # Create tf sparestensor from indices and values to store wd_wts.
        sparse_wts=tf.SparseTensor(indices=input_indices,
                                      values=input_wd_wts,
                                      dense_shape=[self.BATCH_SIZE, vocab_size])

        hidden1=tf.identity(tf.nn.embedding_lookup_sparse(w1, sparse_ids, sparse_wts, combiner = "sum")+b1, name='sparse_hidden1')
        hidden1=tf.nn.dropout(hidden1, self.KEEP_PROB)

        w2 = tf.Variable(tf.random_uniform([SPARSE_HIDDEN_DIM1, num_cates], minval=-1, maxval=1), name='sparse_weights2')
        b2 = tf.Variable(tf.zeros([num_cates]), name='sparse_bias2')
        logits = tf.identity(tf.matmul(hidden1, w2)+b2, name='sparse_logits')
        probs = tf.nn.softmax(logits, name='sparse_probs')

        return w1, w2, logits, probs

    def dnn_train_sparseOnly(self,
                  vecs,
                  targetY,
                  pathname_to_save,
                  SPARSE_HIDDEN_DIM1, 
                  REGULARIZATION=False
                  ):
        """
        The training pipeline, from features to model
        input:
            targetY - target array (sample_size, output_dim)
            pathname_to_save - model path and name
        """

        # Parameters
        output_dim = targetY.shape[1]
        train_steps = int(len(targetY)/self.BATCH_SIZE)+1
        vocab_size = vecs.shape[1]

        # Build Graph
        tf.reset_default_graph()
        y = tf.placeholder(tf.float32, shape=[None, output_dim], name='targets')
        input_indices = tf.placeholder(tf.int64, name='indices') # Indices of nonzero tfidf. Shape = [Unknown,2]
        input_wd_ids = tf.placeholder(tf.int64, name='wd_ids') # wd_ids are simply the second values of indices
        input_wd_wts = tf.placeholder(tf.float64, name='wd_wts') # a wd_wt is the tfidf value at an index
        
        w1, w_out, logits, probs = self.sparse_graph_sparseOnly(input_indices, 
                                         input_wd_ids, 
                                         input_wd_wts, 
                                         vocab_size, 
                                         SPARSE_HIDDEN_DIM1, 
                                         output_dim)

        # Loss function and optimizer
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
        regularizer = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w_out)
        if REGULARIZATION == True:
            loss = tf.reduce_mean(loss+self.BETA*regularizer, name='loss')
        else:
            loss = tf.reduce_mean(loss, name='loss')
        optimizer = tf.train.RMSPropOptimizer(self.LEARNING_RATE).minimize(loss)

        # Run Graph
        saver = tf.train.Saver(max_to_keep=3)
        sess = tf.Session()
        print('Graph started running...')
        print('There will be %d epochs. Each eopch will have %d steps.' %(self.NUM_EPOCHS, train_steps))
        sess.run(tf.global_variables_initializer()) # this could be very slow with large w and large output_dim
        for ep in range(self.NUM_EPOCHS):
            y_batch_gen = self.get_y_batch(targetY, self.BATCH_SIZE)
            sparse_batch_gen = self.get_sparse_batch(vecs, targetY, self.BATCH_SIZE, output_dim)
            total_loss=0.0
            for step in range(train_steps):
                #print((step+1) + train_steps*ep)
                tg_batch = next(y_batch_gen)
                index_batch, wd_wt_batch, _ = next(sparse_batch_gen)
                wd_id_batch = index_batch[:,1]
                loss_batch, _ = sess.run([loss, optimizer],
                                         feed_dict={
                                         y: tg_batch,
                                         input_indices: index_batch,
                                         input_wd_ids: wd_id_batch,
                                         input_wd_wts: wd_wt_batch})
                total_loss += loss_batch
                # Evaluate Training Data
                if ((step+1) + train_steps*ep) % self.EVA_STEP == 0: 
                    print('Average loss at Epoch %d and Step %d is: %f' %(ep, step, total_loss/self.EVA_STEP))
                    total_loss=0.0
                if ((step+1) + train_steps*ep) % self.SAVE_STEP == 0: 
                    saver.save(sess, pathname_to_save, global_step=(step+1) + train_steps*(ep))
                    print("Model saved to: %s" % pathname_to_save)


    def dnn_eval_sparseOnly(self,
                 vecs,
                 targetY,
                 model_path,
                 model_name):
        """
        Restore model
        Run the input data and evaluate accuracy using given labels
        input:
            targetY (list) - known labels
            model_path - tensorflow model path
            model_name - tensorflow model name
        """

        y_batch_gen = self.get_y_batch(targetY, targetY.shape[0])
        tg_batch = next(y_batch_gen)
        sparse_batch_gen = self.get_sparse_batch(vecs, targetY, targetY.shape[0], targetY.shape[1])
        index_batch, wd_wt_batch, _ = next(sparse_batch_gen)
        wd_id_batch = index_batch[:,1]

        # Load Model
        tf.reset_default_graph()
        sess = tf.Session()
        saver = tf.train.import_meta_graph(model_path+model_name)
        saver.restore(sess, tf.train.latest_checkpoint(model_path))

        graph = tf.get_default_graph()
        y = graph.get_tensor_by_name('targets:0')
        input_indices = graph.get_tensor_by_name('indices:0')
        input_wd_ids = graph.get_tensor_by_name('wd_ids:0')
        input_wd_wts = graph.get_tensor_by_name('wd_wts:0')

        probs = graph.get_tensor_by_name('sparse_probs:0')
        loss = graph.get_tensor_by_name('loss:0')

        # Run the model
        feed_dict={
                y: tg_batch,
                input_indices: index_batch,
                input_wd_ids: wd_id_batch,
                input_wd_wts: wd_wt_batch
                }
        test_probs, test_loss = sess.run([probs, loss], feed_dict)
        print("Test loss:", test_loss)
        return test_probs
    
    # Support functions
    def get_dense_batch(self, x, y, batch_size):
        """
        x: 2d array
        y: 2d array
        """

        while True:
            #for i in range(int(x.shape[0]/batch_size)):
            for i in range(int(len(y)/batch_size)):
                data_batch = x[i*batch_size : (i+1)*batch_size]
                tg_batch = y[i*batch_size : (i+1)*batch_size]
                yield data_batch, tg_batch

    def get_sparse_batch(self, vecs, y, batch_size, NUM_CATES):	
            """	
            Text feature generator
            """	
            	
            while True:	
                for i in range(int(len(y)/batch_size)):
                    tg_batch = y[i*batch_size : (i+1)*batch_size]
                    vec_batch = vecs[i*batch_size : (i+1)*batch_size]
                    index_batch = np.array([(el1,el2) for (el1,el2) in zip(*vec_batch.nonzero())])
                    wd_wt_batch = vec_batch.data
                    # If the sparse matrix is too sparse, then some rows can be all zero
                    # so the dim of index_batch will be lower than len(tg_batch) and get error
                    if (len(list(set([id[0] for id in index_batch])))) != batch_size:
                        index_batch = np.array([(el1,el2,el3) for (el1,el2,el3) \
                                           in zip(*vec_batch.nonzero(), vec_batch.data)])
                        missed_row = set(range(batch_size)) - set([id[0] for id in index_batch])
                        for ms_id in missed_row:
                            index_batch = np.append(index_batch, [[ms_id, 0, 0]], axis=0)
                        index_batch_df = pd.DataFrame(index_batch, 
                                                      columns=['row', 'col', 'val'])\
                                            .sort_values(by=['row', 'col'], 
                                                         ascending=[True, False])
                        index_batch=np.array([(el1,el2) for (el1,el2)\
                                      in zip(index_batch_df['row'].astype(int), 
                                             index_batch_df['col'].astype(int))])
    
                        wd_wt_batch = np.array(index_batch_df['val'])
                    yield index_batch, wd_wt_batch, tg_batch

    def get_y_batch(self, y, batch_size):
        """
        y: 2d array
        """

        while True:
            for i in range(int(len(y)/batch_size)):
                tg_batch = y[i*batch_size : (i+1)*batch_size]
                yield tg_batch
                    
          
class CreateFeatures():
    
    def tfidf_fit(self, text_data, pathname_to_save):
            """
            Train tfidf vectorizer and save as a pickle object
            input:
                text_data (series or list) - each element is a string of texts
                pathname_to_save (string) - path and name of vectorizer to be saved
            """
    
            vectorizer=TfidfVectorizer(stop_words='english',
                                       min_df=2,
                                       max_df=0.7,
                                       token_pattern='(?ui)\\b\\w*[A-Za-z]+\\w*\\b')
            vectorizer.fit(text_data)
            pickle.dump(vectorizer, open(pathname_to_save, 'wb'))

    def generate_tfidf_from_text(self,
                                 text_data,
                                 vectorizer_pathname):
        """
        Convert to texts to tfidf vectors using a pre-trained vectorizer
        input:
            vectorizer_pathname(string) - path and name of vectorizer
            data (dataframe) - data contains text fields
            tx_col_names(list) - text column nmaes
        output:
            vecs (sparse matrix) - vectorized list of texts
        """

        print("Load Pre-trained TFIDF Vectorizer")
        vectorizer=pickle.load(open(vectorizer_pathname, 'rb'))
        print("Get The TFIDF of Texts")
        vecs=vectorizer.transform(text_data)

        return vecs