# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains some basic model components"""

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell


class RNNEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional LSTM, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.LSTMCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.LSTMCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("RNNEncoder"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out


class SimpleSoftmaxLayer(object):
    """
    Module to take set of hidden states, (e.g. one for each context location),
    and return probability distribution over those states.
    """

    def __init__(self):
        pass

    def build_graph(self, inputs, masks):
        """
        Applies one linear downprojection layer, then softmax.

        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size)
          masks: Tensor shape (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.

        Outputs:
          logits: Tensor shape (batch_size, seq_len)
            logits is the result of the downprojection layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with vs.variable_scope("SimpleSoftmaxLayer"):

            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None) # shape (batch_size, seq_len, 1)
            logits = tf.squeeze(logits, axis=[2]) # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist


class BasicAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BasicAttn"):

            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output


def masked_softmax(logits, mask, dim):
    """
    Takes masked softmax over given dimension of logits.

    Inputs:
      logits: Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax

    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist


class HeadAttn(object):
    """Module for scaled head attention.

    This only differs from BasicAttn in that it does most of its
    operations with a scaled set of vectors then returns the full size
    weighted sum.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.

    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, scaled_values, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("HeadAttn"):

            # Calculate attention distribution
            values_t = tf.transpose(scaled_values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output


class MultiHeadedAttn(object):
    """Module for multiheaded attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size, num_values, num_heads=8):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size
        self.num_values = num_values
        self.num_heads = num_heads
        self.scaled_attn = [HeadAttn(keep_prob, key_vec_size / num_heads, value_vec_size / num_heads) for _ in range(num_heads)]

    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("MultiHeadedAttn"):


            W_keys = tf.get_variable('W_keys', shape=(self.value_vec_size, self.value_vec_size), initializer=tf.contrib.layers.xavier_initializer())
            shape = keys.get_shape().as_list() + [self.num_heads]
            shape[2] /= self.num_heads
            shape[0] = -1
            scaled_keys = tf.unstack(tf.reshape(tf.tensordot(keys, W_keys, 1), shape), axis=3)
            W_values = tf.get_variable('W_values', shape=(self.value_vec_size, self.value_vec_size), initializer=tf.contrib.layers.xavier_initializer())
            shape = values.get_shape().as_list() + [self.num_heads]
            shape[2] /= self.num_heads
            shape[0] = -1
            scaled_values = tf.unstack(tf.reshape(tf.tensordot(values, W_values, 1), shape), axis=3)

            # shape (batch_size, num_keys, hidden_size, num_heads)
            outputs = tf.stack([self.scaled_attn[i].build_graph(scaled_values[i], values, values_mask, scaled_keys[i])[1] for i in range(self.num_heads)], axis=3)

            # shape (batch_size, num_keys, hidden_size)
            shape = outputs.get_shape().as_list()[0:2] + [self.num_heads*outputs.get_shape().as_list()[2]]
            shape[0] = -1
            return tf.contrib.layers.fully_connected(tf.reshape(outputs, shape=shape), num_outputs=outputs.get_shape().as_list()[2])


class BiDafAttn(object):
    """Module for BiDAF attention.
    """

    def __init__(self, keep_prob, qn_vec_size, context_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          qn_vec_size: size of the question vectors. int
          context_vec_size: size of the context vectors. int
        """
        self.keep_prob = keep_prob
        self.qn_vec_size = qn_vec_size
        self.context_vec_size = context_vec_size

    def build_graph(self, qns, qns_mask, contexts, contexts_mask):
        """

        Inputs:
          contexts: Tensor shape (batch_size, num_contexts, context_vec_size).
          contexts_mask: Tensor shape (batch_size, num_contexts).
            1s where there's real input, 0s where there's padding
          qns: Tensor shape (batch_size, num_qns, context_vec_size)
          qns_mask: Tensor shape (batch_size, num_qns).
            1s where there's real input, 0s where there's padding

        Outputs:
          output: Tensor shape (batch_size, num_qns, 8 * hidden_size).
            This is the attention output
        """
        with vs.variable_scope("BiDafAttn"):

            # Calculate similarity
            num_contexts = contexts.get_shape().as_list()[1]
            num_qns = qns.get_shape().as_list()[1]
            W_sim1 = tf.get_variable('W_sim1', shape=(self.context_vec_size, 1), initializer=tf.contrib.layers.xavier_initializer())
            W_sim2 = tf.get_variable('W_sim2', shape=(self.context_vec_size, 1), initializer=tf.contrib.layers.xavier_initializer())
            W_sim3 = tf.get_variable('W_sim3', shape=(self.context_vec_size, 1), initializer=tf.contrib.layers.xavier_initializer())

            S = []
            c = tf.tensordot(contexts, W_sim1, 1)
            c.set_shape(contexts.get_shape().as_list()[0:2] + [1])
            q = tf.tensordot(qns, W_sim2, 1)
            q.set_shape(qns.get_shape().as_list()[0:2] + [1])
            for j in xrange(num_qns):
                v = tf.tensordot(contexts * tf.expand_dims(qns[:,j,:], 1), W_sim3, 1)
                v.set_shape(c.get_shape())
                S += [c + tf.expand_dims(q[:,j,:], 1) + v]
            S = tf.squeeze(tf.stack(S, axis=2), axis=3) # shape (batch_size, num_contexts, num_qns)

            # Calculate C2Q attention distribution
            qns_attn_logits_mask = tf.expand_dims(qns_mask, 1) # shape (batch_size, 1, num_qns)
            _, qns_attn_dist = masked_softmax(S, qns_attn_logits_mask, 2) # shape (batch_size, num_contexts, num_qns). take softmax over qns

            # Use C2Q attention distribution to take weighted sum of contexts
            a = tf.matmul(qns_attn_dist, qns) # shape (batch_size, num_contexts, qn_vec_size)

            # Calculate Q2C attention distribution
            m = tf.reduce_max(S) # shape (batch_size, num_contexts)
            contexts_attn_logits_mask = tf.expand_dims(contexts_mask, 1) # shape (batch_size, 1, num_contexts)
            _, contexts_attn_dist = masked_softmax(m, contexts_attn_logits_mask, 2) # shape (batch_size, num_contexts). take softmax over contexts

            # Use Q2C attention distribution to take weighted sum of contexts
            c_prime = tf.matmul(contexts_attn_dist, contexts) # shape (batch_size, context_vec_size)

            b = tf.concat([contexts, a, contexts * a, contexts * c_prime], axis=2) # shape (batch_size, num_contexts, context_vec_size*8)

            return b
