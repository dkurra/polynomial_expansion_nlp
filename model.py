import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

BATCH_SIZE = 32


class EquationSolver(keras.models.Model):
    def __init__(self, units=128, encoder_embedding_size=32, decoder_embedding_size=32, input_dim=42, sos_id=43,
                 **kwargs):
        super().__init__(**kwargs)

        # embedding for encoder

        self.sos_id = sos_id
        self.input_dim = input_dim
        """
        The first layer is an Embedding layer, which will convert character IDs into embeddings.
        The embedding matrix needs to have one row per char ID and one column per embedding dimension
        Here we are using embedding dimension of 32
        Whereas the inputs of the model will be 2D tensors of shape [batch size, time steps] i.e. [None, 31], 
        the output of the Embedding layer will be a 3D tensor of shape [batch size, time steps, embedding size] ie. [None, 31, 32]
        """
        self.encoder_embedding = keras.layers.Embedding(input_dim=input_dim + 1, output_dim=encoder_embedding_size)

        """
        encoder is LSTM with 128 units
        return_state=True when creating the LSTM layer so that we can get its final hidden state and 
        pass it to the decoder. Note: LSTM cell has two hidden states (short term and long term)
        """
        self.encoder = keras.layers.LSTM(units, return_sequences=True, return_state=True)

        """
        similar to encoder embedding input [batch_size, time steps] o/p [batch_size, time steps, embedding dimension]
        """
        self.decoder_embedding = keras.layers.Embedding(input_dim=input_dim + 2, output_dim=decoder_embedding_size)

        """
        simply wrap the decoder cell in an AttentionWrapper,
        we provide the desired attention mechanism, we are using Luong attention for this task
        """
        decoder_cell = keras.layers.LSTMCell(units)
        self.luong_attention = tfa.seq2seq.LuongAttention(units)
        self.decoder_cell = tfa.seq2seq.AttentionWrapper(cell=decoder_cell, attention_mechanism=self.luong_attention)

        # output is Dense Layer with input_dim + 1 units
        output_layer = keras.layers.Dense(input_dim + 1)
        # decoder while training
        self.decoder_training_sampler = tfa.seq2seq.sampler.TrainingSampler()
        self.decoder = tfa.seq2seq.BasicDecoder(cell=self.decoder_cell, sampler=self.decoder_training_sampler,
                                                output_layer=output_layer,
                                                batch_size=BATCH_SIZE)

        """
        decoder while inference, 
        almost similar to training decoder except we use GreedyEmbeddingSampler and need to provide maximum_iterations.
        GreedyEmbeddingSampler: 
            computes the argmax of the decoder's outputs and the winner is passed through the decoder_embedding
            Then it is feed to decoder at the next time step
        maximum_iterations: for this task it is set to maximum length of the output sequence in the dataset (29)
        """
        self.decoder_inference_sampler = tfa.seq2seq.sampler.GreedyEmbeddingSampler(embedding_fn=self.decoder_embedding)
        self.inference_decoder = tfa.seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                          sampler=self.decoder_inference_sampler,
                                                          output_layer=output_layer, maximum_iterations=input_dim)

    def call(self, inputs, training=None, **kwargs):
        encoder_input, shifted_decoder_inputs = inputs
        encoder_embeddings = self.encoder_embedding(encoder_input)
        # Note: LSTM cell has two hidden states (short term and long term)
        encoder_outputs, encoder_state_h, encoder_state_c = self.encoder(
            encoder_embeddings,
            training=training)

        encoder_state = [encoder_state_h, encoder_state_c]

        self.luong_attention(encoder_outputs,
                             setup_memory=True)

        decoder_embeddings = self.decoder_embedding(shifted_decoder_inputs)

        decoder_initial_state = self.decoder_cell.get_initial_state(
            decoder_embeddings, batch_size=self.input_dim)
        decoder_initial_state = decoder_initial_state.clone(
            cell_state=encoder_state)

        if training:
            decoder_outputs, _, _ = self.decoder(
                decoder_embeddings,
                initial_state=decoder_initial_state,
                training=training
            )
        else:
            start_tokens = tf.zeros_like(encoder_input[:, 0]) + self.sos_id
            decoder_outputs, _, _ = self.inference_decoder(
                decoder_embeddings,
                initial_state=decoder_initial_state,
                start_tokens=start_tokens,
                end_token=0)

        """
        The decoder outputs (scores) are passed through a softmax
        """
        return tf.nn.softmax(decoder_outputs.rnn_output)
