import numpy as np
import torch

from torch import nn


class OneHotEncoder:
    """
    Class managing the vocabulary and its one-hot encodings
    """
    def __init__(self, vocab):
        """
        Inputs:
            :param vocab (list): List of tokens that can appear in the language
        """
        self.tokens = ["<SOS>", "<EOS>"] + vocab
        self.enc_dim = len(self.tokens)
        self.token_encodings = np.eye(self.enc_dim)

        self.SOS_ENC = self.token_encodings[0]
        self.EOS_ENC = self.token_encodings[1]

        self.SOS_ID = 0
        self.EOS_ID = 1

    def index2token(self, index):
        """
        Returns the token corresponding to the given index in the vocabulary
        Inputs:
            :param index (int)
        Outputs:
            :param token (str)
        """
        return self.tokens[index]

    def enc2token(self, encoding):
        """
        Returns the token corresponding to the given one-hot encoding.
        Inputs:
            :param encoding (numpy.array): One-hot encoding.
        Outputs:
            :param token (str): Corresponding token.
        """
        return self.tokens[np.argmax(encoding)]

    def get_onehots(self, sentence):
        """
        Transforms a sentence into a list of corresponding one-hot encodings
        Inputs:
            :param sentence (list): Input sentence, made of a list of tokens
        Outputs:
            :param onehots (list): List of one-hot encodings
        """
        onehots = [
            self.token_encodings[self.tokens.index(t)] 
            for t in sentence
        ]
        return onehots

    def encode_batch(self, sentence_batch, append_EOS=True):
        """
        Encodes all sentences in the given batch
        Inputs:
            :param sentence_batch (list): List of sentences (lists of tokens)
            :param append_EOS (bool): Whether to append the End of Sentence 
                token to the sentence or not.
        Outputs: 
            :param encoded_batch (list): List of encoded sentences as Torch 
                tensors
        """
        encoded_batch = []
        for s in sentence_batch:
            encoded = self.get_onehots(s)
            if append_EOS:
                encoded.append(self.EOS_ENC)
            encoded_batch.append(torch.Tensor(np.array(encoded)))
        return encoded_batch

    def decode_batch(self, onehots_batch):
        """
        Decode batch of encoded sentences
        Inputs:
            :param onehots_batch (list): List of encoded sentences (list of 
                one-hots).
        Outputs:
            :param decoded_batch (list): List of sentences.
        """
        decoded_batch = []
        for enc_sentence in onehots_batch:
            decoded_batch.append(
                [self.enc2token(enc) for enc in enc_sentence])
        return decoded_batch

class GRUEncoder(nn.Module):
    """
    Class for a language encoder using a Gated Recurrent Unit network
    """
    def __init__(self, context_dim, hidden_dim, word_encoder, 
                 n_layers=1, device='cpu'):
        """
        Inputs:
            :param context_dim (int): Dimension of the context vectors (output
                of the model).
            :param hidden_dim (int): Dimension of the hidden state of the GRU
                newtork.
            :param word_encoder (OneHotEncoder): Word encoder, associating 
                tokens with one-hot encodings
            :param n_layers (int): number of layers in the GRU (default: 1)
            :param device (str): CUDA device
        """
        super(GRUEncoder, self).__init__()
        self.device = device
        self.word_encoder = word_encoder
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            self.word_encoder.enc_dim, 
            self.hidden_dim, 
            n_layers,
            batch_first=True)
        self.out = nn.Linear(self.hidden_dim, context_dim)

    def forward(self, sentence_batch):
        """
        Transforms sentences into embeddings
        Inputs:
            :param sentence_batch (list(list(str))): Batch of sentences.
        Outputs:
            :param unsorted_hstates (torch.Tensor): Final hidden states
                corresponding to each given sentence, dim=(1, batch_size, 
                context_dim)
        """
        # Get one-hot encodings
        enc = self.word_encoder.encode_batch(sentence_batch)

        # Get order of sententes sorted by length decreasing
        ids = sorted(range(len(enc)), key=lambda x: len(enc[x]), reverse=True)

        # Sort the sentences by length
        sorted_list = [enc[i] for i in ids]

        # Pad sentences
        padded = nn.utils.rnn.pad_sequence(
            sorted_list, batch_first=True)

        # Pack padded sentences (to not care about padded tokens)
        lens = [len(s) for s in sorted_list]
        packed = nn.utils.rnn.pack_padded_sequence(
            padded, lens, batch_first=True).to(self.device)

        # Initial hidden state
        hidden = torch.zeros(1, len(sentence_batch), self.hidden_dim, 
                        device=self.device)

        # Pass sentences into GRU model
        _, hidden_states = self.gru(packed, hidden)

        # Re-order hidden states
        unsorted_hstates = torch.zeros_like(hidden_states).to(self.device)
        unsorted_hstates[0,ids,:] = hidden_states[0,:,:]

        return self.out(unsorted_hstates)

    def get_params(self):
        return {'gru': self.gru.state_dict(),
                'out': self.out.state_dict()}

class GRUDecoder(nn.Module):
    """
    Class for a language decoder using a Gated Recurrent Unit network
    """
    def __init__(self, context_dim, word_encoder, n_layers=1, max_length=15,
                 device='cpu'):
        """
        Inputs:
            :param context_dim (int): Dimension of the context vectors
            :param word_encoder (OneHotEncoder): Word encoder, associating 
                tokens with one-hot encodings
            :param n_layers (int): number of layers in the GRU (default: 1)
            :param device (str): CUDA device
        """
        super(GRUDecoder, self).__init__()
        self.device = device
        # Dimension of hidden states
        self.hidden_dim = context_dim
        # Word encoder
        self.word_encoder = word_encoder
        # Max length of generated sentences
        self.max_length = max_length
        # Model
        self.gru = nn.GRU(
            self.word_encoder.enc_dim, 
            self.hidden_dim, 
            n_layers,
            batch_first=True)
        # Output layer
        self.out = nn.Sequential(
            nn.Linear(self.hidden_dim, self.word_encoder.enc_dim),
            nn.LogSoftmax(dim=2)
        )

    def forward_step(self, last_token, last_hidden):
        """
        Generate prediction from GRU network.
        Inputs:
            :param last_token (torch.Tensor): Token at last time step, 
                dim=(1, 1, token_dim).
            :param last_hidden (torch.Tensor): Hidden state of the GRU at last
                time step, dim=(1, 1, hidden_dim).
        Outputs:
            :param output (torch.Tensor): Log-probabilities outputed by the 
                model, dim=(1, 1, token_dim).
            :param hidden (torch.Tensor): New hidden state of the GRU network,
                dim=(1, 1, hidden_dim).
        """
        output, hidden = self.gru(last_token, last_hidden)
        output = self.out(output)
        return output, hidden

    def forward(self, context_batch, target_encs=None):
        """
        Transforms context vectors to sentences
        Inputs:
            :param context_batch (torch.Tensor): Batch of context vectors,
                dim=(batch_size, context_dim).
            :param target_encs (list): Batch of target encoded sentences used
                for teacher forcing. If None then no teacher forcing. 
                (Default: None)
        Outputs:
            :param decoder_outputs (list): Batch of tensors containing
                log-probabilities generated by the GRU network.
            :param greedy_preds (list): Sentences generated with greedy 
                sampling. Empty if target_encs is not None (teacher forcing,
                so we only care about model predictions).
        """
        teacher_forcing = target_encs is not None
        batch_size = context_batch.size(0)

        decoder_outputs = []
        greedy_preds = []
        # For each input sentence in batch
        for b_i in range(batch_size):
            # Initial hidden state
            hidden = context_batch[b_i].view(1, 1, -1)

            # Starting of sentence token
            decoder_input = torch.tensor(
                np.array([[self.word_encoder.SOS_ENC]]),
                device=self.device).float()

            # Maximum number of tokens to generate
            max_l = target_encs[b_i].size(0) if teacher_forcing \
                    else self.max_length

            log_probs = []
            generated_tokens = []
            # For each token to generate
            for t_i in range(max_l):
                # Get prediction
                output, hidden = self.forward_step(decoder_input, hidden)
                # print("OUTPUT", output)

                # Add output to list
                log_probs.append(output.squeeze())

                # Teacher forcing
                if teacher_forcing:
                    # Set next decoder input
                    decoder_input = target_encs[b_i][t_i].view(1, 1, -1)
                else:
                    # Decode output and add to generated sentence
                    # Sample from probabilities
                    _, topi = output.topk(1)
                    # Stop if EOS token generated
                    if topi.item() == self.word_encoder.EOS_ID:
                        break
                    # Save generated token
                    token = self.word_encoder.token_encodings[topi.item()]
                    generated_tokens.append(token)
                    # Set next decoder input
                    decoder_input = torch.Tensor(
                        np.array([[token]]), device=self.device)
            decoder_outputs.append(torch.stack(log_probs))
            greedy_preds.append(generated_tokens)
        
        return decoder_outputs, greedy_preds

    def get_params(self):
        return {'gru': self.gru.state_dict(),
                'out': self.out.state_dict()}
                    