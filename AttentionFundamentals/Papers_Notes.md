# Attention

## 01 - Neural Machine Translation by Jointly Learning to Align and Translate (2015)

### Neural Machine Translation

- **Statistical machine translation** = cooperation of many individually tuned components to achieve scalable machine-to-machine translation
- **Neural machine translation** = single encoder-decoder network architecture that is optimized jointly for the translation task.
  - Encoder embeds original sentence into fixed length vector
  - Decoder outputs a translation from the original encoder
  - Both form a pair that can translate between respective languages, and are optimized jointly

#### Details

Probabilistic interpretation of translation

- $y = \arg\max_\hat{y}{\left( Prob\{\hat{y} | x \}   \right)}$
- Goal = maximize conditional probability
- Train on corpus of translated samples

RNN (e.g. LSTM) encoder-decoder architectures has shown good performance

- RNN Encoder
  - Use the LSTM state to encode words
  - $h_t = f(x_t, h_{t-1})$
  - Compress the states into fixed vector
  - $c = q(\{ h_1, ..., h_T \})$
- RNN Decoder
  - Predict the next word based on LSTM state & previous word & encoder output
  - $p(y_t|\{ y_1, ..., y_{t-1} \}, c) = g(y_{t-1}, s_t, c)$

Alternative architectures with similar goals have also been used.

### Challenge for neural machine translation

- Compression of all useful information into the fixed-length vector
- Long sentences are difficult to compress

### Proposed solution for this problem

- For each generated word
- Determine where the most relevant information resides in the source sentence
- Predict the target word based on these selected relevant words
  = Align & translate

Consequence: Not the whole sentence is encoded into the fixed-length vector, but only the parts relevant for the prediction of the next word

#### Learning to Align & Translate

- The key to enabling alignment is to allow the embedding vector to change with each word
- $p(y_t, \{ y_1, ..., y_{t-1} \}, \vec{x} ) = \text{softmax}(g(y_{t-1}, s_t, c_t))$
- $c_t = \sum_{j=1}^T \alpha_{tj} h_j$
- $\alpha_{tj} = \text{softmax}(a(s_{t-1}, h_j))$
  - Alignment weights can be visualized quite easily to link output words to input word inportance!
- $s$ and $h$ are the internal state variables of the decoder and encoder networks respectively
- By using softmax, the weighting of these variables is normalized!

> [!hint] Note that the alignment is "soft" in the sense that the weighed summation allows smooth propagation of the gradient towards relevant inputs!
> Consequently, the alignment and translation model can be jointly trained.

This mechanism can be interpreted as "attention", allowing the model to tune which aspects of the sentence to consider at any one time.

Consequence: Not the whole sentence is encoded into the fixed-length vector, but only the parts relevant for the prediction of the next word

#### Practicalities

A bidirectional RNN is used

- For each word, the forward and backward hidden states are combined
- This results in a complete picture of the function of that word in the sentence

Tokenization is used to obtain a shortlist of the 30k most used words in each language

#### Results

- Improved performance vs when attention is not used
  - Specifically robustness for longer sentences is drastically improved
- Comparable performance to SoA statistical approach, _if only sentences with known words are considered_

#### Related works

- Graves, 2013 used some version of alignment that increases monotonically, rather than considering the whole sentence at every timestep, in the domain of handwriting generation
- Historically, neural networks were only used as components in a statistical system or to re-rank their generated outputs

## 02 - Effective approaches to attention-based neural machine translation (2015)

### Introduction

- Neural machine translation (NMT)
  - Quickly reached soa performance after introduction
  - Requires minimal domain knowledge
  - Conceptually simple
  - Easy to implement
  - smaller memory footprint than classical machine translation (which require large phrase tables)
- Attention
  - Automated way to figure out alignment between several objects (e.g. images and actions)
  - Enables joint translation and alignment training in NMT

### Challenges

- Attention mechanisms have not yet been sufficiently thoroughly studied
- Standard machine translation typically keeps track of which words it has translated (coverage set). Attentional NMTs should explore similar mechanisms.

### Proposal

Two novel types of attention-based NMT models

- Global = all source words are attended
  - Similar to 01
  - Simpler architecture
    - Use hidden states of a forward LSTM model, rather than a bidirectional RNN
    - Simpler computational path
  - Generalized architecture
    - Only the concat product alignment function was used in 01
    - Extended here
- Local = subset of words are attended
  - Blend of fixed hard & dynamic soft attention models
  - Easier to train than pure hard attention, which requires RL, as it is differentiable
  - Computationally less expensive
  - Attention window $[p_t - D, p_t + D]$
    - Monotonic: $p_t = t$
    - Predictive: $p_t = S \cdot \text{sigmoid}(v_p^T \tanh(W_p h_t))$ = learned window selection strategy
      - Alignment points near $p_t$ are preferred, by enforcing a Gaussian distribution
    - D = hyperparameter
      - Other paper makes this "zoom" parameter dynamic, which complicates everything a lot

Joint attention selection by smart input feeding

- Models should be explicitly aware of previously chosen alignments
- To achieve this, the previous attention vectors are fed along with inputs at each time step.
- The result is a very deep network, both across time (words) and across translation (selection & encoder-decoder)
- **HOW DOES THIS ACTUALLY WORK?**

### Experiments

#### Training

#### English-German Results

- Reversing the source sentence improves the score (1.3)
- Using dropout as well (1.4)
- Global attention boosts performance significantly (2.8)
  - Local attention works even better (3.7)
- Input-feeding helps as well (0.9)
- New SoA model

#### German-English Results

- Similar observations
- interestingly unable to beat SoA here

### Analysis

- Very similar trends to 01 displayed when it comes to robustness to long sentences.
- The improvement of this paper vs 01 is rather incremental, but it is still worth it to see how to extract more performance out of this idea
- Explicit evaluation of the alignment error rate (AER) is performed

## 03 - Attention Mechanisms in Neural Networks - Where it comes and where it goes (2022)

### Introduction

- Attention
  - Mechanism inspired by the human visual system
    - We don't see the entire world at once
    - Only small patch is in focus
    - Saccades = eye movement to scan the environment until sufficient information is available for recognition
    - Processing is dynamically restricted to a subset of the visual field
    - 2 Questions: What to look at & Where to look!
  - Attempts made to transfer such a mechanism to AI tasks
  - Remarkable performance has been achieved using this mechanism

### History of attention

#### 1980s to 2000s

- Idea of attention has drifted around in some form since the 1980s
- (1980-1997) First models focus on visual areas and attention scanning
- Region of (1994) interest extraction or (1991) target detection are some explicit attempts at focusing the "attention" of processing in images
- (1995) Systems doing partial analysis and combining results afterwards were also proposed
- (1995) Selective (input sensitivity) tuning is proposed

#### 2000s to 2010s

Focus on making attention mechanisms more useful for neural networks

- (2001) Networks with seperate "where" and "what" pathways for object detection
- (2002) Neural networks & markov models combined for object detection
- (2005) RL techniques such as Q-learning are proposed to train attention in real-world situations
- (2006) Models simulating human eye movements are experimented with for class detection
- (2006) Models are trained to detect the most relevant parts of color pictures
- (2007) Subject tracking in videos using attention mechanisms
- Many of these models focus on image classification
- (2014) DasNet: a deep neural network with recursive connections learned through reinforcement leaning is proposed for selective attention to certain features
- (2014) Attention-based framework for generative modelling is proposed

#### 2015: The rise of attention

- 2015 = golden year of attention mechanisms
- Three initial studies sparked an avalanche
- 1. Neural machine translation using encoder-decoder models with attention (01)
  - Avoid the need of compressing all information of long sentences into the fixed-length intermediary representation
  - Annotations of a BiRNN $a_j$ are weighed by energy factors $e_{ij}$, computed by a feed-forward neural network. These energy factors are normalized by the softmax function to $\alpha_{ij}$, which can in turn be interpreted as relative importances.
  - The annotations are then weighted by their relative imortances to generate the context vector $c_i = \sum_{j=1}^T{\alpha_{ij} a_j}$
  - The result is some form of "soft attention" applied to the input sequence for the generation of each new word
- 2. Attention applied to visual system for image captioning
  - CNN used as encoder to encode the image
  - RNN used as decoder to generate the caption
  - $a_j$ in this case corresponds to CNN output channel slices (???)
  - The way in which weights $\alpha_{ij}$ are computed is very similar
  - The way in which they are combined, denoted $\phi$ has two variants
    - Hard (stochastic) -> Trainable by reinforcement learning
      - Define $s_t$ = location variable, which is used to decide where to focus attention
      - $\alpha_{ij}$ is interpreted as a Bernoulli probability that decides whether $a_j$ is selected $p(s_t = 1 | a_j) = \alpha_{ij}$
      - Effectively this is equivalent to a stochastic discretization / binarization of the soft method
      - Both stochastic & deterministic versions have the same expected value of outcome
    - Soft (deterministic) -> Trainable by backpropagation
- 3. Neural machine translation with local & global attention
  - Two kinds of attention
    - Global attention
      - Soft attention that always pays attention to all source words
      - Differences with (1)
        - A score function is used, rather than the original energy definition
        - Different versions of the score function are considered
    - Local attention
      - Subset of words are selected using hard attention mechanisms, which are relatively weighed using soft attention
      - Aligned position $p_t$ in the source sentence is generated for each generated word
      - The subset of words is a selection around $p_t$
- The way in which these 3 papers introduced attention made an actual and significant difference for neural network performance

#### 2015-2016: Attack of the attention

Attention mechanisms are introduced in many applications

- Memory networks
  - (Store and retrieve information from a memory matrix)
  - Neural turing machines allow for end-to-end training of such networks with explicit memory cells
  - End-to-end memory networks advance upon this by adding a recurrent attention mechanism
- Self-attention in LSTM networks
  - Modify LSTM structure by replacing the memory cell with a memory network (key vectors -> value vectors)
  - As such, memory and attention are added within the LSTM sequence encoder network, rather than added on top of it (- TODO: read paper to truly grasp this)
  - The self-attention model can relate different poisitions of its inputs
- Other applications - image captioning - abstractive summarization - speech recognition - automatic video captioning - neural machine translation - recoginizing textual entailment - visual question answering - modelling sentence pairs - video classification
  At the same time, new neural network architectures are introduced
- Stacked attention network (image question answering)
- deep attention recurrent Q-network (soft & hard attention in deep Q-networks)
- wake-sleep recurrent attention model (faster image classification & captioning training)
- alignDRAW (deep recurrent attention writer) (generative image-from-caption model)
- generative adversarial what-where network (GAN with attention - instructions contain what to draw in what location)
