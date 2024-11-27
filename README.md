# NEATJulia

[![Build Status](https://github.com/Rithvik-Reddy-Adapa/NEATJulia.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Rithvik-Reddy-Adapa/NEATJulia.jl/actions/workflows/CI.yml?query=branch%3Amain)

*!! A hobby project built from scratch, can have bugs 🪲. !!*

Inspired from the concept of [NEAT (Neuro-Evolution of Augmenting Topologies)](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf), i.e., evolving nature of Neural Network (NN) using Genetic Algorithm (GA).

## Features
- __Multi-Threaded__ for better performance.
- Supports both __FFNNs__ (Feed-Forward Neural Networks) & __RNNs__ (Recurrent Neural Networks).
- Supports both __Forward Connections__ & __Backward Connections__.
- Support for __LSTM__ (Long-Short Term Memory) & __GRU__ (Gated Recurrent Unit) nodes under RNNs.
- __Log to file__ & __Log to console__.
- Ability to __save__ & __load__ entire NEAT struct or just a/specific Network(s). Ability to save NEAT struct every *n* iterations of training.
- Can export dot language code to __Visualize Network__. Can also export to *.svg*, but requires `dot` executable available in the path.
- You can have different mutation probabilities for different types of nodes and connections.

## References

- __Research 🎓__
  - OG paper on [NEAT](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
- __Youtube 🎞️__
  - [NEAT - Introduction](https://www.youtube.com/watch?v=VMQOa4-rVxE) - [Finn Eggers](https://www.youtube.com/@finneggers6612)
  - [Snake learns with NEUROEVOLUTION (implementing NEAT from scratch in C++)](https://www.youtube.com/watch?v=lAjcH-hCusg) - [Tech With Nikola](https://www.youtube.com/@TechWithNikola) (Nikola)
  - [NEAT Algorithm Visually Explained](https://www.youtube.com/watch?v=yVtdp1kF0I4) - [David Schäfer](https://www.youtube.com/@DavidSchaeferDataScience) (David Schaefer)
  - [How to train simple AIs to balance a double pendulum](https://www.youtube.com/watch?v=9gQQAO4I1Ck) - [Pezzza's Work](https://www.youtube.com/@PezzzasWork) (John Buffer)
- __Github 🧑‍💻__
  - [Neat-Python](https://github.com/CodeReclaimers/neat-python) - Code Reclaimers
  - [NEAT.jl](https://github.com/Andy-P/NEAT.jl) - Andy P
  - [Neataptic](https://github.com/wagenaartje/neataptic) - Thomas Wagenaar
  - [Evolvable RNN](https://github.com/RubenPants/EvolvableRNN) - Ruben Pants
- __Web 🌐__
  - [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) - Christopher Olah

Refer to the 2 examples provided for better understanding on how to use the code.
