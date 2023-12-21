# Deep Learning for Music Genre Classification

## Overview
This project applies deep learning to enhance music genre classification, a key aspect for digital platforms like Spotify. We use the GTZAN dataset and transform 30-second audio clips into spectrograms. Our method employs Densely Connected Convolutional Networks (DenseNet) and adaptive training techniques to improve accuracy and prevent overfitting.

## Approach
Our approach to music genre classification is innovative, using CNN models and new data preprocessing techniques. We combine Mel and STFT spectrograms for data augmentation, addressing the challenge of limited labeled audio data.

## Implementation
We preprocess the GTZAN dataset into STFT and Mel spectrograms for a comprehensive audio representation. We start with a basic CNN model, then advance to DenseNet121, experimenting with both pre-trained and non-pretrained versions.

## Experimentation 
We use both STFT and Mel spectrograms from the GTZAN dataset, combining these two formats to capture a comprehensive representation of audio for CNN application in music genre classification.**

## Results and Conclusions
Our findings demonstrate the effectiveness of DenseNet architectures coupled with dual-spectrogram data augmentation. We highlight the importance of model complexity, data augmentation, and pretraining in enhancing model performance.

### Performance Metrics
- **Baseline CNN Model**: Achieved an AUROC score of 0.75 on the test set.
- **DenseNet121 (Non-pretrained)**: Improved the AUROC score to 0.82 on the test set.
- **DenseNet121 (Pretrained)**: Further improved the AUROC score to 0.85 on the test set.

### Confusion Matrix Analysis
The confusion matrix revealed that our model performed best on genres such as Classical and Jazz, with F1-scores of 0.89 and 0.87 respectively. However, it struggled with more similar genres like Rock and Metal, which had F1-scores of 0.78 and 0.76 respectively.

### Comparative Analysis
Comparing the DenseNet models, the pretrained version outperformed the non-pretrained version by 3% in terms of AUROC score. This suggests that pretraining provides a useful inductive bias for this task, despite the differences between music spectrograms and natural images.

### Future Work
We aim to further improve the model's performance by exploring other architectures like ResNets and EfficientNets, and experimenting with larger and more diverse datasets. We also plan to investigate other data augmentation techniques and their impact on model performance. 
