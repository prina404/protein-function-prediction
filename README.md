# Experimental Transformer branch 

This branch contains a failed experiment which consisted in training a transformer from scratch.


Major headaches:

- Vanilla transformer with quadratic attention is basically infeasible on single GPU, unless context window is kept at a ridiculously small size. 
- Using padding is WAY easier and effective than using nested tensors

Note for future me: use padding and linear/sparse attention    