# BAN6420_M6
Module 6 Assignment: Fashion MNIST Classification
Here's a README.md file that explains the steps and details of the project:


# Fashion MNIST CNN Model
This project aims to build a Convolutional Neural Network (CNN) to classify images from the Fashion MNIST dataset. The Fashion MNIST dataset consists of 60,000 28x28 grayscale images of 10 different clothing categories, such as t-shirts, trousers, dresses, and more. This model uses various techniques like data augmentation, batch normalization, dropout, and early stopping to optimize the performance of the model.

 Table of Contents
  - Setup
  - Model Architecture
  - Training
  - Results
  - How to use


 1. Setup
     - Clone this repository to your local machine.
        git clone https://github.com/AmazingTaiwo/BAN6420_M6.git
     - Install the required Python libraries.
        pip install -r requirements.txt
   
        The requirements.txt file includes:
        - tensorflow
        - numpy
        - matplotlib
        - keras
     You can also install the libraries individually if you prefer.

3.  Model Architecture

    The model consists of the following layers:
    - Convolutional Layer 1: 32 filters with a kernel size of (3, 3), ReLU activation, and batch normalization.
    - Max Pooling Layer 1: Pooling size of (2, 2).
    - Convolutional Layer 2: 64 filters with a kernel size of (3, 3), ReLU activation, and batch normalization.
    - Max Pooling Layer 2: Pooling size of (2, 2).
    - Convolutional Layer 3: 128 filters with a kernel size of (3, 3), ReLU activation, and batch normalization.
    - Dropout Layer: Dropout rate of 40% to reduce over-fitting.
    - Flatten Layer: To flatten the output from the convolutional layers.
    - Fully Connected Layer: 128 neurons with ReLU activation and batch normalization.
    - Output Layer: 10 neurons for the 10 classes of Fashion MNIST, using a softmax activation function.

4.  Training
   The model was trained using the following settings:
        - Optimizer: Adam with a learning rate of 0.0005
        - Loss Function: Sparse categorical cross-entropy
        - Metrics: Accuracy
        - Batch Size: 64
        - Epochs: 20
        - Callbacks:
            - Early Stopping: Monitors validation loss and stops training if the loss doesn't improve for 3 consecutive epochs.
            - Model Checkpoint: Saves the best model based on validation accuracy.

    Additionally, data augmentation was applied to the training images:
     - Rotation: Random rotations up to 20 degrees
     - Width & Height Shift: Random shifts by up to 20%
     - Zoom: Random zoom by up to 20%.
     - Horizontal Flip: Random horizontal flipping

6.  Results
   The model achieved a test accuracy of approximately X% on the Fashion MNIST test set (this value can be determined after running the model). After training, the model's performance was plotted for both training and validation accuracy, as well as training and validation loss.

    # Example of Predictions:
        Below is an example of predictions on test images:

        1. Image 1:
            - Predicted: T-shirt
            - Actual: T-shirt

        2. Image 2:
            - Predicted: Trouser
            - Actual: Trouser

# How to Use(Python Script):
    1.  Run the script ban6420_m6_3.py to train and evaluate the model on cmd or python IDE.
    2.  This will will automatically load the Fashion MNIST dataset, apply data augmentation, and train the CNN model.
# How to use (Rscript):
      - Open Rstudio
      - Run the script ban6420_m6_rstudio.R to train and evaluate the model using Rstudio.
      - This will will automatically load the Fashion MNIST dataset, apply data augmentation, and train the CNN model.
