
The vanilla Transformer is a sequence-to-sequence model typically used for Machine translation and consists of an encoder and a decoder, each of which is a stack of $N$ identical blocks. The encoder maps an input sequence of symbol representations $(x_1, ..., x_n)$ to a sequence of continuous representations $\mathbf{z} = (z_1, ..., z_n)$. Given $\mathbf{z}$, the decoder then generates an output sequence $(y_1, ..., y_m)$ of symbols one element at a time. At each step the model is auto-regressive, consuming the previously generated symbols as additional input when generating the next. Basically, the encoder first encodes the full sentence, then the decoder decodes one word at time.

- **Encoder:** The encoder is composed of a stack of $N$[^1] identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network. For building a deeper model, a residual connection is employed around each module, followed by Layer Normalization module.
    
- **Decoder:** Compared to the encoder blocks, decoder blocks additionally insert cross-attention modules over the output of the encoder stack between the multi-head self-attention modules and the position-wise FFNs. Furthermore, the self-attention modules in the decoder are adapted to prevent each position from attending to subsequent positions.
    

[^1]: $N=6$ in the original paper

Transformer is a model that uses **attention** to boost the speed. More specifically, it uses self-attention. Transformer allows for significantly more parallelization, since the tokens can be processed independently.

![[dl-trans-arch.png]]

## Input Embedding

An embedding is a vector that **semantically** represents an object/input. In the context of NLP, the goal is to transform a text (set of words) into a vector of numbers such that similar words produce similar vectors. The **word2vec** technique is based on a feed-forward, fully connected architecture. It is similar to an autoencoder, but rather than performing input reconstruction, word2vec trains words according to other words that are neighbors in the input corpus.

Word2vec can learn word embedding in two different ways:

- **CBOW** (Continuous Bag of Words) in which the neural network uses the context to predict a target word.
    
- **Skip-gram** in which it uses the target word to predict a target context.
    

![[dl-word2vec-env.png]]

## Positional Encoding

Position and order of words are the essential parts of any language. They define the grammar and thus the actual semantics of a sentence. Recurrent Neural Networks (RNNs) inherently take the order of word into account. They parse a sentence word by word in a sequential manner. This will integrate the words' order in the backbone of RNNs.

Since the model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence.

One possible solution to give the model some sense of order is the **positional encoding**.

Positional encoding describes the location or position of an entity in a sequence so that each position is assigned a unique representation. There are many reasons why a single number, such as the index value, is not used to represent an item's position in transformer models. For long sequences, the indices can grow large in magnitude, thus causing problem with the optimization phase. Trying to normalize the index value to lie between 0 and 1, could create problems for variable length sequences as they would be normalized differently (i.e. position 1 in a sequence of length 10 would have an encoding 10 times smaller in magnitude for a sequence of length 100).

Transformers use a smart positional encoding scheme, where each position/index is mapped to a vector. Hence, the output of the positional encoding layer is a matrix, where each row of the matrix represents an encoded object of the sequence summed with its positional information.

Suppose you have an input sequence of length $L$. The positional encoding is given by sine and cosine functions of varying frequencies:

$$ \begin{align} PE_{(k, 2i)} &= \sin(k/n^{2i/d}) \\
 PE_{(k, 2i + 1)} &= \cos(k/n^{2i/d}) \end{align} $$

where:

- $k$ is the position of an object in the input sequence.
- $d$ is the dimension of the output embedding space.
- $PE_{(k,j)}$ is the position function for mapping a position $k$ in the input sequence to index $j$ of the positional matrix.
- $n$ is a user-defined scalar, set to 10000 by the authors of the original Transformer's paper.
- $i$ is used for mapping to column indices $0 \leq i < d/2$, with a single value of $i$ maps to **both** sine and cosine functions.

You can also imagine the positional embedding as a vector containing pairs of sines and cosines for each frequency.

![[dl-pos-encoding1.png]]

The scheme for positional encoding has a number of advantages:

- The sine and cosine functions have values in $[-1, 1]$, which keeps the values of the positional encoding matrix in a normalized range.
    
- As the sinusoid for each position is different, you have a unique way of encoding each position.
    
- You have a way to add positional information to words embedding in order to encode the relative positions of words (e.g. the words 'brother' and 'sister' will probably have similar embedding representations, but if one word is used in a very different position than the other, they may not be correlated. In order to get different representations, we add positional information).
    

The positional encoding layer sums the positional vector with corresponding word embedding vector.

![[dl-pos-encoding2.png]]

## Attention

In order for the decoding to be precise, it needs to take into account every word of the input, using attention.

An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.
This resembles finding a value in a hash table, but instead of having a "hard" match on a single key, a "matching score" is attributed to all the keys in the table, which influences the values that get more weight to compute the output of the query.

![[dl-attention-arch.png]]

The first step in calculating attention is to create three vectors from each of the encoder's input vectors. So for each word, we create a **Query vector**, a **Key vector**, and a **Value vector**. These vectors are created by multiplying the embedding by three matrices ($\mathbf{W}^{\mathbf{Q}}, \mathbf{W}^{\mathbf{K}}, \mathbf{W}^{\mathbf{V}}$) that are composed of trainable parameters, to be obtained during training.

The $\mathbf{W}^{Q}$, $\mathbf{W}^{K}$ and $\mathbf{W}^{V}$ matrices have size $d \times d_{k}$, with $d$ being the size of the embeddings input sequence and $d_{k}$ being the size of the keys/values internal representation.
Then $\mathbf{Q} = \mathbf{X}\mathbf{W}^{Q},\mathbf{K} = \mathbf{X}\mathbf{W}^{K}, \mathbf{V} = \mathbf{X}\mathbf{W}^{V}$ (in the original paper the input sequence embeddings are the _rows_ of the input matrix).

![[dl-attention-arch2.png]]

If we collect all the query, keys and value vectors in three matrices ($\mathbf{Q}, \mathbf{K}, \mathbf{V}$), we can define the attention function as follows:

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

By taking the dot product between the query matrix and the key matrix, we compute a _score_ for each word of the input sentence against the others. The score determines how relevant is a word with respect to the others. The dot-products of queries and keys are divided by $\sqrt{d_k}$ (dimensionality of the embedding vector) to alleviate gradient vanishing problem of the softmax function. The softmax normalizes the scores so they're all positive and add up to 1. $\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$ is often called **attention matrix**.

Then, we compute the dot product between the attention matrix and the value matrix $\mathbf{V}$. The intuition here is to keep intact the values of the relevant word(s), and drown-out irrelevant words (by multiplying them by tiny numbers in the attention matrix).

### Multi-Head Attention

Instead of simply applying a single attention function, Transformer uses **multi-head attention**, where queries keys and values are linearly projected $h$ times with different, learned linear projections to $d_k$, $d_k$ and $d_v$ dimensions, respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding $d_v$-dimensional output values. These are concatenated and once again projected via $\mathbf{W}^{O}$ to obtained the desired output representation $\mathbf{Y}$. Notice that, in order to be able to normalize the results, $\mathbf{Y}$ must have the same size as the input.

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.

$$\text{MultiHead}(Q, K, V ) = \text{Concat}(\text{head}_1, ..., \text{head}_h)\mathbf{W}^{\mathbf{O}}$$

where

$$\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_i^{\mathbf{Q}}, \mathbf{K}\mathbf{W}_i^{\mathbf{K}}, \mathbf{V}\mathbf{W}_i^{\mathbf{V}})$$

![[dl-multihead-attention.png]]

### Add & Norm and Feed Forward Network

In addition to attention sub-layers, each of the layers in the encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between.

$$\text{FFN}(h) = \max(0, hW_1 + b_1)W_2 + b_2$$

where $h$ is the output of the previous layer.

In order to build a deep model, Transformer employs a residual connection around each module, followed by Layer Normalization. For instance, each Transformer encoder block may be written as

$$ \begin{align} \mathbf{H'} &= \text{LayerNorm}(\text{SelfAttention}(\mathbf{X}) + \mathbf{X})\\ \mathbf{H} &= \text{LayerNorm}(\text{FFN}(\mathbf{H'}) + \mathbf{H'}) \end{align} $$

![[dl-transformer-norm.png]]

### Stacking Encoding Blocks

We can stack together multiple encoding blocks, each taking in input the output of the previous block in the chain.
Due to this the embedding and positional encoding is only applied once before the first encoder block, since after that we already are in embeddings' space.

![[dl-transformer-stack.png]]

### Decoding

In the decoding phase, the final representation from the last encoding block is fed into the decoder's cross-attention layers. In cross-attention, the decoder constructs queries from its own hidden state using its learned $\mathbf{W}^Q$ matrix, while the keys and values are constructed from the encoder's output using the decoder's own $\mathbf{W}^K$ and $\mathbf{W}^V$ matrices. This allows the decoder to 'query' the encoded representation - essentially asking 'what information from the input sequence is relevant to what I'm currently generating?' The encoder's output serves as a memory that the decoder can selectively attend to, with the attention mechanism determining which parts of the encoded input are most relevant for generating each output token.

At the beginning of the decoder a mask is applied to prevent leftward illegal information flow, so that it cannot use information from the future during training, which would be invalid during inference.

![[dl-transformer-decoding.png]]

### Applications of Attention in the Model

The Transformer uses multi-head attention in three different ways:

- In "encoder-decoder attention" layers (**cross-attention**), the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence.
    
- The encoder contains **self-attention** layers. In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder stack. we set $\mathbf{Q} = \mathbf{K} = \mathbf{V} = \mathbf{Z}$, where $\mathbf{Z}$ is the output of the previous encoder layer.
    
- Masked Self-attention: We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. Therefore, in the Transformer decoder, the self-attention is restricted such that queries at each position can only attend to all key-value pairs up to and including that position. This is implemented inside of scaled dot-product attention by masking out (setting to $-\infty$) all values in the input of the softmax which correspond to illegal connections.

![[dl-transformer-crossatt.png]]
    

