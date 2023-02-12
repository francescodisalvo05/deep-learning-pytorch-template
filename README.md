# Deep Learning PyTorch template

Over the last years I've built several deep learning projects in PyTorch, either for work or for academic projects. What I realized was that I was actually writing and writing the same things, except for a few things. Therefore, in order to avoid writing always the same pipeline, I wrote my own deep learning template in PyTorch! 🚀

## Structure

```
└─── assets/ : store results, images or major checkpoints
|
└─── dataset/ : store the dataset involved on your tests
|
└─── src/ 
└───────── data/ 
└───────────────── dataset.py : dataset class
└───────────────── transformations.py : data augmentation pipeline
└───────── losses/
└───────── metrics/ 
└───────── models/
└───────────────── model.py : custom model
└───────────────── trainer.py : trainer class
└───────── utils/ 
└───────────────── plots.py : plots sample of images, metrics and so on
└───────────────── utils.py : set seed, early stopping and so on
|
└─── .gitignore
└─── README.md
└─── settings.yaml : settings for the current experiment 
└─── train.py : training pipeline with logging, cuda and tensorboard support
```

This is still a work-in-progress template and I would be happy to receive comments and feedback based on your professional experience! 

> Inspired by [PyTorch-Deep-Learning-Template](https://github.com/FrancescoSaverioZuppichini/PyTorch-Deep-Learning-Template)

### To do:

- [ ] Create a project based on this template
- [ ] Add more info about the code and what it is supported so far
