# GloVe-word-sum-demo

## This project uses pre-trained word vectors from https://www-nlp.stanford.edu/projects/glove/ to find the nearest neighbour for the sum of two or more words.

### WARNING: this repo only contains the first few entries in glove.6B.50d.txt!

### To compile: 

##### run `rustc main.rs`

### To run:

##### run `./main <glove_vectors_filename.txt> word1 word2 word3 ...`

### Example:

##### `./main glove.6B.50d.txt grimace shake`
##### `Loading GloVe vectors...`
##### `Nearest neighbor: shaking (similarity: 0.7648)`


