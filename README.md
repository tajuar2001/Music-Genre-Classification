# Music Genre Classification Using Deep Learning

## Abstract
This project leverages the advancements in deep learning for improving music genre classification, crucial for digital platforms like Spotify. Using the GTZAN dataset, we transform 30-second audio clips into Short-time Fourier Transform (STFT) and Mel-scale spectrograms. Our approach utilizes Densely Connected Convolutional Networks (DenseNet) and introduces adaptive training techniques to enhance classification accuracy and prevent overfitting, employing AUROC metrics for comprehensive model evaluation.

## Introduction
Music genre classification is a subjective and complex challenge gaining traction in machine learning. We explore an innovative approach to data augmentation in music genre classification using CNN models, focusing on overcoming the limitations posed by the scarcity of labeled audio data.

## Related Work
Our research builds upon existing methods in music genre classification, employing spectrogram data and exploring new data preprocessing techniques. We introduce a novel method combining Mel spectrograms with STFT spectrograms as data augmentation, setting our approach apart from traditional methods.

## Methodology
### Data Preprocessing
We use both STFT and Mel spectrograms from the GTZAN dataset, combining these two formats to capture a comprehensive representation of audio for CNN application in music genre classification.

### Model Architecture
We start with a basic CNN model and then advance to the DenseNet121 architecture. We experiment with both pre-trained and non-pretrained versions of DenseNet121 to evaluate the effectiveness of our dual-spectrogram technique.

## Experiments
### Dataset
Our primary dataset is the GTZAN dataset, containing 10 distinct music genres. The dataset is split into 80% training, 10% validation, and 10% testing sets.

### Hyperparameters and Model Configuration
We use the Adam optimizer with a learning rate of 0.0005 and Cross-Entropy Loss. Our training strategy includes early stopping and model checkpointing.

### Evaluation Metrics
Our models are evaluated using the AUROC metric and a confusion matrix to understand per-class performance.

### Comparative Model Analysis
We compare the performance of different model configurations, including baseline CNN and various DenseNet models with and without dual-spectrogram augmentation.

## Results and Conclusions
Our findings demonstrate the effectiveness of DenseNet architectures coupled with dual-spectrogram data augmentation. We highlight the importance of model complexity, data augmentation, and pretraining in enhancing model performance.
