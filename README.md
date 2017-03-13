First of all, I am not in the course officially. I am just doing it because it sounds like a cool project.  

# Captions
There has been some incredible results recently for filling images, but none of them were using captions. This is why I focus mainly on trying to introduce captions in recent similar model. 

I am spending most of my time on this project trying to find a good ways to incorporate the information contained in the caption into the generative model. 

I hesitated between different techniques, but I realised that I firstly need to find good word embedding or train a model based on all captions. 

There are at least 5 captions per images, each caption has ~10 words. It seems like a too small corpus to train the model. 

## Pre-trained Stanford embedding
I've used Stanford trained Glove embedding for a while, so I think it could be a nice idea to re-used them. I look for the proportion of word in the caption vocabulary that have a pre-trained embedding.... only one half. Not enough to capture the meaning of the sentences.  
![Pre trained model](http://nlp.stanford.edu/projects/glove/)

## FastText
FAIR releases recently pre-trained embedding for different languages. Embedding are trained on all Wikipedia. _Fasttext_ can be a good choice because it works with words that are not in the initial training vocabulary. I've never read the paper of the algorithm (don't read, don't trust), and I've used them for other tasks, but I always found Glove embedding were of better quality, leading better model. Maybe someone could try to used them and see if they perform well.  
![Pre trained model](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md)

## Skip-Thought vectors
Quoted from the abstract of the paper _"We describe an approach for unsupervised learning of a generic, distributed sentence encoder. Using the continuity of text from books, we train an encoder-decoder model that tries to reconstruct the surrounding sentences of an encoded passage. Sentences that share semantic and syntactic properties are thus mapped to similar vector representations"_. It means like a nice approach to try because we have different sentences meaning the same thing, so every caption should be transformed in vector that are closed in the high dimensional space.   
![Pre trained model](https://github.com/ryankiros/skip-thoughts).  
It takes quite a while to create the embedding, and I am still generating some of them.

To assess the quality of the embedding created, I created a little script to plot embedding. I consider only the first $\frac{6000 captions}{5 caption per images}$ images.

### Initial guess
I am hopping that sentences which have relative same meaning, should be closed in space. Because the vector generated is 4800 long, I needed a manifold learning algorithm. In Tensorflow, it is possible to used PCA, or t-SNE to visualize them in 3 dimensions. 
Another way, without plotting them, would be to do a nearest neighbor search. 

Here are some results (code is in _helper\_visualize.py_, make sure you create a metadata.tsv file before containing captions):  
(Notes: Every caption has an embedding, which means that every image appears five times in the plot, I am hopping they are close).
#### Similar captions in PCA

* I was searching for **computer** and pick the first caption _"A cat laying on top of a computer keyboard_. I restrict the point in space to the closest 100 points. It seems that similar caption contains whether a computer, whether a cat, or a keyboard, which are all part of the caption... Good start.
<details> 
  <summary>PCA plot 1</summary>
   ![](images/computer_embedding.PNG)
</details>

* While training the model for filling holes, I was amazed by how elephants where nicely fit, so I looked at closed captions of _"A herd of elephants standing next to each other"_. The computed nearest neighbors are elephant, giraffe, or zebras and others...
<details> 
  <summary>PCA plot 2</summary>
    ![](images/elephant.PNG)
</details>

#### Exploring the points with t-SNE
* Picture of people skiing  
<details> 
  <summary>t-SNE plot 1</summary>
    ![](images/snow_images.PNG)
</details>

* London red bus (or bus in general)  
<details> 
  <summary>t-SNE plot 2</summary>
    ![](images/red_bus.PNG)
</details>

__Take home message:__
I think embedding trained with the Skip-Though vector model have captured some interesting semantic of the captions but ....

### Are they influencing the model
See model section to see the current model used. 
No, not at all. To prove that, I did some experiments. 

I tried three experiments. __This ten images have never been seen by the model__, __I re-run every experiments twice__:
* Replace embedding by random noise (noise is sample from a Gaussian distribution(0; 0.7))
<details> 
  <summary>First experiment results</summary>
    ![](images/first_training_example_with_false_embeddings.PNG)
    ![](images/second_training_example_with_fake_embeddings.PNG)
</details>


* True sampling from the Gaussian distribution with mean and variance coming from the fully connected.
<details> 
  <summary>Second experiment results</summary>
    ![](images/first_training_example_with_good_embeddings.PNG)
    ![](images/second_training_example_with_good_embeddings.PNG)
</details>

* Replace averaging embedding by selecting only one embedding
<details> 
  <summary>First experiments results</summary>
    ![](images/not_mean_caption.PNG)
</details>


I really wanted to say "Yes embedding makes the difference", but as far as I know, I don't see any differences. And if I see some differences sometimes, I guess it's because I want to see them...  
I have some insights why the model didn't need them, but I am open to discuss this (StackGAN results are very domain specific: drawing birds or flower.).  
<details> 
  <summary></summary>
    ![](images/powerful.jpg)  
</details>

# Model
## A. Queues and data augmentation
I am using Tensorflow queues, because it is much simpler than feeding a dictionary, way faster, allow me to fastly experimenting new model, and I can make some data augmentation before passing data to the model.  
Right now I am only randomly flipping right or left the images. I am planning to add some techniques explained in this [blog](http://machinelearningmastery.com/image-augmentation-deep-learning-keras/).  
Also, i don't fill the image with black or white color, I am filling every pixel to fill in each channel with the respective average channel color of the cropped image.   
I crop the image in the middle, for simplicity. Some papers claimed it does not harm generalization (but I am not sure if it's more than they were lacking time for experimenting it :) ).

## B. Embedding embedding in the model
tldr: I used a technique I found in the recent StackGan architecture. They advanced some arguments about conditioning a model with high dimensional vectors:
* "However, latent space conditioned on text is usually high dimensional (> 100 dimensions). With limited amount of data,  it usually causes discontinuity in the latent data manifold, which is not desirable for learning the generator."
* "To further enforce the smoothness over the conditioning manifold and avoid overfitting, we add the following regularization term to the objective of the generator during training".. KL divergence. 

After seeing the results from the paper, I was convinced it will help the "generative" model (in the sense of generating images) towards the tasks of filling images.
Here is the plot of the KL loss. During the first iterations, I saw it going down, but afterwards it restart to going up. Note that the large increase of the curve represents the moment when I introduced new examples to the model, so I guess the KL divergence is responsible of avoiding some overfitting.
<details> 
  <summary>KL loss plot</summary>
    ![](images/kl_loss.PNG)  
</details>

Before sampling from a normal distribution, embedding, five in total per images, are averaged. I did this, because based on the previous embedding plots, I think averaging embeddings should give me the a point in the high dimension space which has some notions of the semantic from every captions. 
__Notes__: I know here are some more advanced technique such as using a recurrent neural network, but in this configuration, i don't think averaging should harm the training model.  

The average of all captions is then passed into a fully connected, which output a mean and (log) variance. Using the reparametrization-trick on a Gaussian(0, 1), it is possible to sample from this distribution. A vector of dimension 100 is sampled from this distribution. It should represent the embedding. This technique became popular with variationnal neural auto encoder. 

## C. Auto-Encoder
### 1. Architecture
I did a pretty deep neural network (~30 layers), but it's basically a encoder-decoder with a lot of residual blocks in the decoder. I didn't use anything special between the encoder and the encoder, some people are using channel wise, I think passing the channel is better. The embedding is stacked on top of convolution channels (4x4x512 + 4x4x100). An embedding is replicated in all 4x4 dimensions. 
Code is in ```model.py```

#### Miscellaneous
<details> 
  <summary>Batch normalization all the way long</summary>
    ![](images/batch_norm.jpg)  
</details>

* dropout every three layers
* batch size is small (gpu is never fully loaded), but I found experimentally helping the optimization process.  

### 2. Loss function
Classic L2 loss, but for smoothness in the border, I encourage more pixel at the frontier to be closed the truth. When seeing the images generated, I saw the difference of this trick: it really remove the effect of a squared block append on top of the image, which is pretty cool

### 3. Adversarial cost
There are out two papers who claimed to remove the blurriness in generated images.
* One is using neural style transfer, and it takes three minutes to generate a new images. With the recent progress in Neural Style Transfer [google paper](https://arxiv.org/pdf/1610.07629v2.pdf), maybe
* The other one, wait for it... is a GAN which try to differentiate between truth generated image and fake ones. The D should have some insights of what is a correct images with wrong captions, so the D should consider truth images, with wrong captions as fake sample. [paper](https://arxiv.org/abs/1604.07379).  
<details> 
  <summary></summary>
    ![](images/meme.jpeg)  
</details>

I don't have a long experience with GAN's but every time I wanted to train them, I had to stay close my computer, because it is very unstable, and whether the G doesn't learn anything, whether G and D loss keeps oscillating.  
    <details> 
      <summary></summary>
        ![](images/training_gan.jpg)  
    </details>

For this project, I decided to give a shot to the recent WGAN. As far as I understood, the paper claims we can train discriminator to convergence, then train the generator on it. 
    <details> 
    <summary></summary>
        ![](images/d_optimum.jpg)  
    </details>
  
In compensation, gradients of the discriminator need to be clipped to small values... but we don't need a sigmoid at the end of the D, which is also responsible for vanishing the gradient. 
* The generator loss become the previous loss defined in 2) + ```tf.reduce_mean(-(features_fake_image + feature_fake_captions)```
* The discriminator loss is ```tf.reduce_mean(features_fake_image + feature_fake_captions - features_real)```. 
As of now, it does not help the model, and I observed that the inside generated images become completely off-context (It fill the images with a squared blurry images which does not have the same color).  
I need to fix it.  

### Regularization
I added dropout as a regularizer in the middle of the training.  
#### Test set images
I didn't found any improvement on unseen pictures:
<details> 
   <summary>Truth image</summary>
    ![](images/dp_truth_image.PNG)  
</details>
<details> 
  <summary>With dropout</summary>
    ![](images/wt_dropout.PNG)  
</details>
<details> 
  <summary>Without dropout</summary>
    ![](images/without_dropout.PNG)  
</details>

#### Training set images
However, I think it helps the model to not overfit on the training set, as you can see on this images. Without dropout, it's difficult to see the frontier between the holes fitted and the border, which is not true with dropout.
<details> 
   <summary>Truth image</summary>
    ![](images/dp_truth_image_train.PNG)  
</details>
<details> 
  <summary>With dropout</summary>
    ![](images/wt_dropout_train.PNG)  
</details>
<details> 
  <summary>Without dropout</summary>
    ![](images/without_dropout_train.PNG)  
</details>
### Preliminary results
* Loss functions:  
    <details> 
    <summary>All loss functions</summary>
        ![](images/loss_f.PNG)
    </details>
* Sample images unseen (always generated then true pictures)
    <details>
      <summary>First sample</summary>
        ![](images/sample1_fake.PNG)
        ![](images/true_image1.PNG)
    </details>
    <details>
      <summary>Second sample</summary>
        ![](images/sample2_fake.PNG)
        ![](images/true_image2.PNG)
    </details>
    <details>
      <summary>Third sample</summary>
        ![](images/sample3_fake.PNG)
        ![](images/true_image3.PNG)
    </details>
    <details>
      <summary>Fourth sample</summary>
      ![](images/more_examples.jpeg)
    </details>
    <details>
      <summary>Fifth sample</summary>
        ![](images/sample4_fake.PNG)
        ![](images/true_image4.PNG)
    </details>

I observed that it's harder for the model to fill images when the background is very blurry in its nature, like vegetation.  
 
# Clame
* I write as I think, and my english is far from perfect, sorry if reading this is hurting your eyes.  
* Contribute or ask questions in _Issues_, if you want :)
* TfRecord files are quite large, but I can share them.  

# RoadMap
- [X] Finish generating embeddings
- [ ] Reduce learning rate before overfitting
- [X] Add decaying dropout to the training
- [ ] Is DRAW outdated based on GAN recent results... DRAW architecture were every iteration is condition on previous generated images + new caption?
- [ ] Better data augmentation
- [ ] Make GAN works, whether WGAN or classical ones
- [ ] Refactor the code
- [ ] Try to generate "HD" images, like in StackGAN (need to retrieve from MS-COCO for supervised learning)

