Previous architectures
- LSTM
- Gated RNNs

**Self attention** - Relates different positions of a single sequence in order to compute a representation of the sequence

Input - $(x_1, \ldots, x_n)$
Encoding - $\mathbf{z} = (z_1, \ldots, z_n)$
Output - $(y_1, \ldots, y_m)$

Transduction models have encoder/decoder structure
- encoder maps inputs $x$ to representation $z$ 
- decoder generates $y$ given $z$

**Autoregresssive** - consumes the previously generated symbol as additional input

Transformer architecture

![[Screenshot 2026-08-15 at 22.50.20.png|536]]
- Decoder uses masked self-attention to prevent positions from attending to subsequent positions.
	- Ensures prediction for position $i$ can only depend on known outputs at positions less than $i$

**Scaled dot-product attention**
- Queries and keys of dimension $d_k$, values of dimension $d_v$
$$
\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$
**Multi-head attention**
- Instead of $d_{model}$ dimensional keys for queries, values and keys, split them $h$ times, perform attention parallely and then concatenated and projected
- This allows the model to attend information from different representations at different positions

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
$$
where
- $\text{head}_i = \text{Attention}(QW^Q_i, KW_i^K, VW_i^V)$
- $W_i^Q \in \mathbb{R}^{d_\text{model} \times d_k}$, ... and $W^O \in \mathbb{R}^{hd_v \times d_\text{model}}$

Attention uses in Transformers
- Encoder-decoder attention layers - queries come from previous decoder and keys, values comes form the output of encoder
	- Allows each position in decoder to attend over all positions of input
- Encoder's self attention - each position in encoder can attend to all positions in previous layer of encoder
- Decoder self attention - uses masked attention so that each position can only attend to positions up to and including that position

**Feed forward networks**
$$
FFN(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

**Softmax**
- Use a learnt linear transformation and softmax to turn the decoder output to predicted next-token probabilities. 

**Positional encoding**
- Injects information about the position of the tokens in the sequence
