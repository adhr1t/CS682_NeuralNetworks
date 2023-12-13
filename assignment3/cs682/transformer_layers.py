import torch
import torch.nn as nn
from torch.nn import functional as F
import math

"""
This file defines layer types that are commonly used for transformers.
"""

class PositionalEncoding(nn.Module):
    """
    Encodes information about the positions of the tokens in the sequence. In
    this case, the layer has no learnable parameters, since it is a simple
    function of sines and cosines.
    """
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        """
        Construct the PositionalEncoding layer.

        Inputs:
         - embed_dim: the size of the embed dimension
         - dropout: the dropout value
         - max_len: the maximum possible length of the incoming sequence
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert embed_dim % 2 == 0
        # Create an array with a "batch dimension" of 1 (which will broadcast
        # across all examples in the batch).
        pe = torch.zeros(1, max_len, embed_dim)
        ############################################################################
        # TODO: Construct the positional encoding array as described in            #
        # Transformer_Captioning.ipynb.  The goal is for each row to alternate     #
        # sine and cosine, and have exponents of 0, 0, 2, 2, 4, 4, etc. up to      #
        # embed_dim. Of course this exact specification is somewhat arbitrary, but #
        # this is what the autograder is expecting. For reference, our solution is #
        # less than 5 lines of code.                                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # create indexes for length dimension. Called i as in the equation
        len_i_dim = torch.arange(start = 0, end = max_len).reshape(-1, 1) # reshape so will be (l, 1)
        # create indexes for embed dimension. Called j as in the equation
        # step = 2 for even exponents up till embed_dim
        embed_j_dim = torch.arange(start = 0, end = embed_dim, step = 2) # (1, d)
        
        # assign pe with sin(math.pow(i*10000, -(j/embed_dim))) if j is even
        # assign pe with cos(math.pow(i*10000, -((j-1)/embed_dim))) otherwise
        
        # if I just feed embed_j_dim into the equation, the exponents are already even
        # so I don't need to do any checking there bc the equation's (j-1) takes care
        # of the odd values of j too
        # (l, 1) * (1, d) = (l, d)

        # print(math.pow(10000, -(embed_j_dim/embed_dim))) # error broadcasting
        # when using math.pow

        # print(len_i_dim * (10000 ** -(embed_j_dim/embed_dim)))
        pos_enc_sin = torch.sin(len_i_dim * (10000 ** -(embed_j_dim/embed_dim)))
        pos_enc_cos = torch.cos(len_i_dim * (10000 ** -(embed_j_dim/embed_dim)))
        
        
        # need to assign these values in pe
        # values need to be assigned in dimensions that will be added to X embedding dimension
        # even_nos = list(filter(lambda x: (x % 2 == 0), list1))
        pe[:, :, ::2] = pos_enc_sin # assign to even values
        pe[:, :, 1::2] = pos_enc_cos # assign to odd values

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # Make sure the positional encodings will be saved with the model
        # parameters (mostly for completeness).
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Element-wise add positional embeddings to the input sequence.

        Inputs:
         - x: the sequence fed to the positional encoder model, of shape
              (N, S, D), where N is the batch size, S is the sequence length and
              D is embed dim
        Returns:
         - output: the input sequence + positional encodings, of shape (N, S, D)
        """
        N, S, D = x.shape
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, S, D))
        ############################################################################
        # TODO: Index into your array of positional encodings, and add the         #
        # appropriate ones to the input sequence. Don't forget to apply dropout    #
        # afterward. This should only take a few lines of code.                    #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pe_select = self.pe[:N, :S, :D]
        output = x[:, :, :] + pe_select
        # error saying dimension 1 don't match so cant add
        # restrict all dimensions of pe to be N, S, D

        # dropout
        output = self.dropout(output)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return output


class MultiHeadAttention(nn.Module):
    """
    A model layer which implements a simplified version of masked attention, as
    introduced by "Attention Is All You Need" (https://arxiv.org/abs/1706.03762).

    Usage:
      attn = MultiHeadAttention(embed_dim, num_heads=2)

      # self-attention
      data = torch.randn(batch_size, sequence_length, embed_dim)
      self_attn_output = attn(query=data, key=data, value=data)

      # attention using two inputs
      other_data = torch.randn(batch_size, sequence_length, embed_dim)
      attn_output = attn(query=data, key=other_data, value=other_data)
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Construct a new MultiHeadAttention layer.

        Inputs:
         - embed_dim: Dimension of the token embedding
         - num_heads: Number of attention heads
         - dropout: Dropout probability
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        # We will initialize these layers for you, since swapping the ordering
        # would affect the random number generation (and therefore your exact
        # outputs relative to the autograder). Note that the layers use a bias
        # term, but this isn't strictly necessary (and varies by
        # implementation).
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        self.attn_drop = nn.Dropout(dropout)

        self.n_head = num_heads
        self.emd_dim = embed_dim
        self.head_dim = self.emd_dim // self.n_head

    def forward(self, query, key, value, attn_mask=None):
        """
        Calculate the masked attention output for the provided data, computing
        all attention heads in parallel.

        In the shape definitions below, N is the batch size, S is the source
        sequence length, T is the target sequence length, and E is the embedding
        dimension.

        Inputs:
        - query: Input data to be used as the query, of shape (N, S, E)
        - key: Input data to be used as the key, of shape (N, T, E)
        - value: Input data to be used as the value, of shape (N, T, E)
        - attn_mask: Array of shape (S, T) where mask[i,j] == 0 indicates token
          i in the source should not influence token j in the target.

        Returns:
        - output: Tensor of shape (N, S, E) giving the weighted combination of
          data in value according to the attention weights calculated using key
          and query.
        """
        N, S, E = query.shape
        N, T, E = value.shape
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, S, E))
        ############################################################################
        # TODO: Implement multiheaded attention using the equations given in       #
        # Transformer_Captioning.ipynb.                                            #
        # A few hints:                                                             #
        #  1) You'll want to split your shape from (N, T, E) into (N, T, H, E/H),  #
        #     where H is the number of heads.                                      #
        #  2) The function torch.matmul allows you to do a batched matrix multiply.#
        #     For example, you can do (N, H, T, E/H) by (N, H, E/H, T) to yield a  #
        #     shape (N, H, T, T). For more examples, see                           #
        #     https://pytorch.org/docs/stable/generated/torch.matmul.html          #
        #  3) For applying attn_mask, think how the scores should be modified to   #
        #     prevent a value from influencing output. Specifically, the PyTorch   #
        #     function masked_fill may come in handy.                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        H = self.n_head

        # reshape from (N, T, E) into (N, T, H, E/H)
        # query, key, value are all derived from X
        query_out = self.query(query)
        query_reshape = query_out.reshape(N, S, H, E//H) # N, S, E -> N, S, H, E//H
        # N, S, H, E//H -> N, H, S, E//H
        query_reshape = query_reshape.permute(0, 2, 1, 3)

        key_out = self.key(key)
        key_reshape = key_out.reshape(N, T, H, E//H) # N, T, E -> N, T, H, E//H
        # N, T, H, E//H -> N, H, T, E//H
        key_reshape = key_reshape.permute(0, 2, 1, 3)

        value_out = self.value(value)
        value_reshape = value_out.reshape(N, T, H, E//H) # N, T, E -> N, T, H, E//H
        # N, T, H, E//H -> N, H, T, E//H
        value_reshape = value_reshape.permute(0, 2, 1, 3)


        ### put the Multi-Headed Scaled Dot-Product Attention function into code
        
        # dot-product attention: torch.matmul Q and K transpose
        dot_prod_attn = torch.matmul(query_reshape, key_reshape.permute(0, 1, 3, 2)) 
        # (N, H, S, E//H) * (N, H, E//H, T) = (N, H, S, T)
        # (N, H, T, E/H) by (N, H, E/H, T) to yield a shape (N, H, T, T)
        
        # scale by dividing sqrt(E//H)
        dot_prod_attn = dot_prod_attn/math.sqrt(E//H)


        #attn_mask needed for Masked Multi-Head attention
        if attn_mask != None: # is not None
            dot_prod_attn = dot_prod_attn.masked_fill(attn_mask == 0, float('-inf')) 
            # setting all values of 0 to -infinity so output is not influenced
            # bc mask value is so tiny
            
        # Softmax on Target Sequence Length
        dot_prod_attn_softmax = F.softmax(dot_prod_attn, dim = 3)

        # Dropout
        dot_prod_attn_dropout = self.attn_drop(dot_prod_attn_softmax)
        
        # (N, H, S, T) * (N, H, T, E//H) = (N, H, S, E//H)
        Y = torch.matmul(dot_prod_attn_dropout, value_reshape)
        
        ## linear transformation of the concatenation of the heads
        # reshape output to linear
        # Transposing the structure so it's (N, S, H, E//H)
        # then recombine H and E//H into E by reshaping
        Y = Y.permute(0, 2, 1, 3)
        Y = Y.reshape(N, S, E)
        
        
        output = self.proj(Y)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return output


