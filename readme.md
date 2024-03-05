This text file contains the task provided and later details of the implementation (inclduing what each file does)
As a coding task, I would suggest that the following then makes the most sense:

1) Reimplement the gradient accumulation method from this paper: https://arxiv.org/abs/2307.09542. This is supposed to find out which neurons are most influenced by a data point during training.
2) Train a VGG7 model with the MNIST dataset. As done in the paper, assign random labels to 10% of the data points from MNIST. These are the outliers.
3) Then, for the 10% outliers, use gradient accumulation to find out which neurons they have the largest influence to.

4) Take 10% of the normal data point and use gradient accumulation to find which neurons they have the largest influence to.

5) Report and visualize with a visualization method of your choice, how large the overlap between the neurons is. Also see if you can find other interesting insights, like, for example, if the memorized data points are mainly memorized by the same neurons, or all by different ones – what their class has got to do with it. Potentially, it will also make sense to visualize the data points for that.

### Implementation

* preparedata.py:
    Contains a function to add noise to the dataset and return the indices of those examples

* train_outliers.ipynb:
    trains a VGG-7 model on the MNIST dataset containing 60,000 examples with 10% noisy examples 
    Noise: The output has been changed randomly for 6,000 examples
    The model is trained on 54,000 + 6,000 clean and noisy examples
    The resulting model is also saved using checkpoints. these are the .ckpt files
    The indices of noisy samples is saved in "noisy_y_train.csv", "noisy_indices.csv" to continue working on them for locating important neurons

* get_neurons.ipnyb:
    Reload original model, data, noisy and clean examples and work with samples of size 1000
    For each example in noisy_sample and clean_sample obtain the individual gradients (saved as .pkl files)
    Then compute batch_gradients
    Using argmax, get the important neuron per layer for each example
    For each example there are 6 neuron (i.e. most important neuron per layer for that example)

    Returns files examples_to_neurons_clean.pkl  and examples_to_neurons_noisy.pkl 

* evaluate_neurons.ipynb:
    Reload original model, data, noisy and clean samples, and their gradients 
    Zero out the most important neuron until prediction changes. Save index and #neurons needed for flipping
    Finally evaluate:
        The number of neurons needed for flipping predictions
        Overlap of neurons for clean and noisy examples per layer
        Memorization depending on the labels.

