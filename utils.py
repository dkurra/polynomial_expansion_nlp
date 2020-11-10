import tensorflow as tf
import numpy as np


def process_result(r):
    return r.replace(' ', '')

def ragged_tensor(mat):
    return tf.ragged.constant(np.array(mat)).to_tensor()


def prepare_new_sequences(factors, tokenizer, max_input_length=29, max_output_length=28):
    # prepares new sequence to convert into train data format, pls refer to expand notebook for more explanation
    seqs = tokenizer.texts_to_sequences(factors)
    X_new = ragged_tensor(seqs)

    if X_new.shape[1] < max_input_length:
        X_new = tf.pad(X_new, [[0, 0], [0, max_input_length - X_new.shape[1]]])
    X_decoder = tf.zeros(shape=(len(X_new), max_output_length), dtype=tf.int32)
    return X_new, X_decoder


def chunks(lst, n):
    # yeilds a list of size n
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

