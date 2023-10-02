<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CIFAR-10 Training Repository Documentation</title>
</head>
<body>

<h1>CIFAR-10 Training Repository Documentation</h1>

<h2>Overview</h2>
<p>This repository contains scripts to train models on the CIFAR-10 dataset using PyTorch. The primary script, <code>main.py</code>, provides a comprehensive workflow from loading CIFAR-10 data, setting up models, and training them using various hyperparameters and optimization strategies.</p>

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
