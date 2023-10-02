<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<h1>CIFAR-10 Training Repository Documentation</h1>

<h2>Overview</h2>
<p>This repository contains scripts to train models on the CIFAR-10 dataset using PyTorch. The primary script, <code>main.py</code>, provides a comprehensive workflow from loading CIFAR-10 data, setting up models, and training them using various hyperparameters and optimization strategies.</p>

<p>The model used for training in this repository is the Network In Network (NiN) model, based on <a href="https://arxiv.org/abs/1312.4400" target="_blank">this paper</a>.</p>

<img src="https://github.com/VenturaHaze/CIFAR10_Pytorch/blob/993c01031709d2ceb06b4a5d57c0a185823f30f8/CIFAR-10-visualization.png" alt="CIFAR-10-Visualization">

<h2>Dataset Classes</h2>

<p>The CIFAR-10 Database provided by Toronto University<a href="https://www.cs.toronto.edu/~kriz/cifar.html" target="_blank">is accessible via this link.</a>
<p>The dataset contains the 10 following classes:</p>
<ul>
    <li>Airplane</li>
    <li>Automobile</li>
    <li>Bird</li>
    <li>Cat</li>
    <li>Deer</li>
    <li>Dog</li>
    <li>Frog</li>
    <li>Horse</li>
    <li>Ship</li>
    <li>Truck</li>
</ul>

<h3>Sample Images After Transformations</h3>
<img src="https://github.com/VenturaHaze/CIFAR10_Pytorch/blob/dd3dec5c9de3f58e1ee9438fd5c17569414f88da/sample_transformations.png" alt="Sample CIFAR-10 Images After Transformations">
<p>This image showcases a sample of CIFAR-10 images after applying the data transformations used in the training process.</p>

<h2>Repository Structure</h2>
<p>The repository is structured with the following main scripts:</p>
<ul>
    <li><strong>data_setup.py</strong>: Handles data preparation and loading processes for CIFAR-10.</li>
    <li><strong>engine.py</strong>: Contains core functionalities for the training loop of the classifier.</li>
    <li><strong>eval.py</strong>: Manages model evaluation processes.</li>
    <li><strong>main.py</strong>: The primary script for executing the model training workflow.</li>
    <li><strong>metrics.py</strong>: Defines various metrics used for evaluating model performance.</li>
    <li><strong>models.py</strong>: Provides model architectures and related utilities.</li>
</ul>

<h2>Repository Structure</h2>
<p>The repository is structured with the following main scripts:</p>
<ul>
    <li><strong>data_setup.py</strong>: Handles data preparation and loading processes for CIFAR-10.</li>
    <li><strong>engine.py</strong>: Contains core functionalities for the training loop of the classifier.</li>
    <li><strong>eval.py</strong>: Manages model evaluation processes.</li>
    <li><strong>main.py</strong>: The primary script for executing the model training workflow.</li>
    <li><strong>metrics.py</strong>: Defines various metrics used for evaluating model performance.</li>
    <li><strong>models.py</strong>: Provides model architectures and related utilities.</li>
</ul>

<h2>Key Components</h2>

<h3>main.py</h3>
<p>The primary script, <code>main.py</code>, is structured with the following segments:</p>
<ol>
    <li><strong>Imports</strong>: Necessary packages and libraries are imported including PyTorch, NumPy, and Matplotlib.</li>
    <li><strong>Device Setup</strong>: The script detects and sets up CUDA if available.</li>
    <li><strong>Data Loading</strong>: 
        <ul>
            <li>Parameters related to the data loading process, such as batch size, augmentation, and validation fraction, are set up.</li>
            <li>Data loaders for the CIFAR-10 dataset are created with the specified parameters.</li>
            <li>The classes in the dataset are displayed.</li>
            <li>A sample batch of images is visualized to inspect the dataset.</li>
        </ul>
    </li>
    <li><strong>Model Setup</strong>: 
        <ul>
            <li>Model parameters are specified including the model name (like 'NiN') and whether to use a pretrained model.</li>
            <li>The chosen model is loaded and inspected using <code>torchinfo</code>.</li>
        </ul>
    </li>
    <li><strong>Optimizer & Scheduler Setup</strong>: 
        <ul>
            <li>The optimizer (SGD or Adam) and learning rate scheduler are chosen and set up based on the provided parameters.</li>
        </ul>
    </li>
    <li><strong>Training</strong>: 
        <ul>
            <li>Directories for saving the trained model are created or checked.</li>
            <li>The training process is initiated using the <code>train_classifier_simple_v2</code> function from the <code>engine</code> module.</li>
            <li>Optionally, the script provides an alternative training process using <code>train_classifier_simple_v1</code>.</li>
            <li>After training, logs are saved as pickle files for future reference.</li>
        </ul>
    </li>
</ol>

<h2>Auxiliary Modules</h2>
<ol>
    <li><strong>data_setup</strong>: This module is assumed to contain functions for setting up dataloaders, specifically <code>get_dataloaders_cifar10</code> for the CIFAR-10 dataset.</li>
    <li><strong>models</strong>: This module provides choices for different models, and the required resolution for the chosen model is determined using the <code>model_choice</code> function.</li>
    <li><strong>engine</strong>: This module is a central component of this repository, providing core functionalities for the training loop of the classifier. Key features include:
        <ul>
            <li>Train and evaluation loops for the classifier.</li>
            <li>Logging capabilities to monitor loss and accuracy metrics over epochs.</li>
            <li>Support for various loss functions and optimization strategies.</li>
        </ul>
    </li>
</ol>

<h2>Getting Started</h2>
<p>To run the training script:</p>
<pre><code>python main.py</code></pre>
<p>This will execute the entire workflow as defined in <code>main.py</code>, from loading data, setting up the model, to training.</p>

<h2>Conclusion</h2>
<p>This repository provides a straightforward way to train models on the CIFAR-10 dataset using PyTorch. With the modular setup and commented sections in the <code>main.py</code> script, it offers flexibility for experimentation with different models, optimizers, and training configurations.</p>

</body>
</html>
