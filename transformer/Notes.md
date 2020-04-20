Output: producing a classification
The most common way to build a sequence classifier out of sequence-to-sequence layers, is to apply global average pooling to the final output sequence, and to map the result to a softmaxed class vector.


Overview of a simple sequence classification transformer. 
The output sequence is averaged to produce a single vector representing the whole sequence. 
This vector is projected down to a vector with one element per class and softmaxed to produce probabilities.

