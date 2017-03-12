# Captions
I've been spending quite a long time trying to find a good ways to incorporate the informations contained in the caption into the generative model.  
I hesitated between different techniques, but I realised that I firstly need to find good word embeddings or train a model based on all captions.  
There are at least 5 captions per images, each captions has ~10 words. It seems like a too small corpus to train the model.  
### Pre-trained Stanford embeddings
I've been using stanford trained Glove embedding, so I try to used them. I look for the proportion of word in the caption vocabulary that have a pre-trained embedding.... only one half. Not enought to capture the meaning of the sentences.  
![Pre trained model](http://nlp.stanford.edu/projects/glove/) 

### FastText
FAIR releases recently pre-trained embedding for different languages. Embeddings are trained on all Wikipedia. _Fasttext_ can be a good choice because it works with words that are not in the initial training vocabulary. I've never read the paper of the algorithm (don't read, don't trust), and I've used them for other tasks, but I always found Glove embeddings where of better quality, leading better model. Maybe someone could try to used them.  
![Pre trained model](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md)

### Skip-Thought vectors
Quoted from the abstract of the paper _"We describe an approach for unsupervised learning of a generic, distributed sentence encoder. Using the continuity of text from books, we train an encoder-decoder model that tries to reconstruct the surrounding sentences of an encoded passage. Sentences that share semantic and syntactic properties are thus mapped to similar vector representations"_. It means like a nice approach to try because we have different sentences meaning the same thing, so every caption should be transformed in vector that are closed in the high dimensionnal space. I used an already trained model from ![here](https://github.com/ryankiros/skip-thoughts)

