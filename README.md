# State of my art results:
## Training set
![](images/new_running_experiment.PNG)
![](images/new_running_experiment2.PNG)

![](images/new_running_experiment3.PNG)  
and the true image, to see the difference
![](images/new_running_experiment3_true.PNG)  

## Testing set:
* Generated image: 
![](images/new_running_experiment_test_fake.PNG)  
* True image:
![](images/new_running_experiment_test_true.PNG)  

* Generated image:
![](images/new_running_experiment_test_fake2.PNG)  
![](images/new_running_experiment_test_fake3.PNG)  
* True image:
![](images/new_running_experiment_test_true3.PNG) 

# What worked for me (and what didn't)
_See Thoughts.md_

# RoadMap
- [X] Finish generating embeddings
- [X] Reduce learning rate before overfitting
- [X] Add decaying dropout to the training
- [ ] Is DRAW outdated based on GAN recent results... DRAW architecture were every iteration is condition on previous generated images + new caption?
- [X] Better data augmentation
- [ ] Make GAN works, whether WGAN or classical ones
- [ ] Refactor the code
- [X] Try to generate "HD" images, like in StackGAN (need to retrieve from MS-COCO for supervised learning)
- [ ] Used ideas from there: ![](https://ppaquette.github.io/)
