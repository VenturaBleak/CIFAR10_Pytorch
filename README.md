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

![Model Output Overlayed with Image]([https://github.com/VenturaHaze/Solar_France/blob/b10afc533674a4496456138f3c04eb34fa8ca861/UNet_pretrained100_Epoch10_pred1.png](https://github.com/VenturaHaze/CIFAR10_Pytorch/blob/993c01031709d2ceb06b4a5d57c0a185823f30f8/CIFAR-10-visualization.png))



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
    <li><strong>engine</strong>: This module provides two versions of training functions: <code>train_classifier_simple_v1</code> and <code>train_classifier_simple_v2</code>.</li>
</ol>

<h2>Getting Started</h2>
<p>To run the training script:</p>
<pre><code>python main.py</code></pre>
<p>This will execute the entire workflow as defined in <code>main.py</code>, from loading data, setting up the model, to training.</p>

<h2>Conclusion</h2>
<p>This repository provides a straightforward way to train models on the CIFAR-10 dataset using PyTorch. With the modular setup and commented sections in the <code>main.py</code> script, it offers flexibility for experimentation with different models, optimizers, and training configurations.</p>

</body>
</html>
