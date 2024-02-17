# Practical Work in AI: Optimizing SNN for Face Recognition

This repository showcases the implementation and findings of a comprehensive study focused on optimizing Siamese Neural Networks (SNN) for face recognition tasks. The study, spearheaded by Boran Cihan Polat, delves into the nuances of neural network architectures, particularly SNN, and evaluates their performance across various loss functions and activation functions.
![GitHub Image](/figure5.png)

## Overview

The project investigates the application of different loss functions (Binary Cross-Entropy Loss, Contrastive Loss, and Triplet Loss) and activation functions (ReLU and SeLU) within the SNN framework to identify the most effective combinations for face recognition tasks.

## Key Insights

- **Loss Functions:** The comparative analysis revealed that Triplet Loss significantly enhances SNN's ability to distinguish between faces.
- **Activation Functions:** The study highlights the superior performance of the SeLU activation function over ReLU, attributing its effectiveness to its self-normalizing properties.
![GitHub Image](/figure4.png)
- **Hyperparameter Tuning:** An exploration into hyperparameter tuning's impact on SNN's accuracy and efficiency, suggesting avenues for further optimization.

## Findings

Our experiments underscore the importance of carefully selecting and tuning the loss and activation functions to improve face recognition accuracy. Triplet Loss emerged as a highly effective loss function, with SeLU providing notable improvements in network performance due to its self-normalizing feature.

### Experimental Results

Included in our analysis is an insightful plot (Figure 6) illustrating the performance comparison across different loss functions and activation functions within the SNN framework. This visualization underscores the nuanced impacts these factors have on the overall effectiveness of face recognition models.

![GitHub Image](/figure6.png)

*Figure 6: Performance comparison of SNN across different loss functions and activation functions.*

## Usage

- Clone the repository.
- Install dependencies as outlined in `requirements.txt`.
- Follow the provided documentation to replicate the study or apply the findings to your own face recognition projects.

## Future Directions

The findings from this study open several avenues for future research, including further optimization of hyperparameters and exploration of additional neural network architectures to enhance face recognition capabilities.

## Acknowledgments

This work is based on the study "Practical Work in AI" by Boran Cihan Polat. For a deeper dive into the methodology, results, and theoretical underpinnings, refer to the original study.
