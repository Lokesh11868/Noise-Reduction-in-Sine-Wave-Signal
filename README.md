# Noise Reduction in Sine Wave Signal

## Project Overview
This project demonstrates how to denoise a noisy sine wave signal using deep learning models. The primary models include:
- **Conv1D Autoencoder**
- **LSTM Autoencoder**
- **Enhanced Conv1D Autoencoder**

Each model takes a noisy signal as input and reconstructs the original (denoised) signal. Additionally, the **Savitzky-Golay filter** is applied for further smoothing.

## Methodology
### Preprocessing:
- The signal is **normalized** using MinMaxScaler.
- Both noisy and original signals are **reshaped into sliding windows** for training.

### Model Architectures:
- **Conv1D Autoencoder:** Uses convolutional and pooling layers to extract features and reduce noise.
- **LSTM Autoencoder:** Uses LSTM layers to capture **temporal dependencies**, making it effective for time-series data.
- **Enhanced Conv1D Autoencoder:** A deeper architecture with additional layers for better reconstruction.

### Postprocessing:
- The **Savitzky-Golay filter** is applied to reduce residual noise further.

### Visualization:
- The **original, noisy, and denoised signals** are plotted for comparison.

## Dependencies
- Python 3.x
- pandas
- numpy
- scikit-learn
- tensorflow
- scipy
- matplotlib

## Model Details
### Conv1D Autoencoder
A convolutional model for feature extraction and denoising:
```text
Layer (type)                    Output Shape         Param #
=================================================================
input_signal (InputLayer)        [(None, 4500, 1)]    0
conv1d (Conv1D)                  (None, 4500, 16)     64
max_pooling1d (MaxPooling1D)     (None, 2250, 16)     0
conv1d_1 (Conv1D)                (None, 2250, 8)      392
max_pooling1d_1 (MaxPooling1D)   (None, 1125, 8)      0
conv1d_2 (Conv1D)                (None, 1125, 8)      200
up_sampling1d (UpSampling1D)     (None, 2250, 8)      0
conv1d_3 (Conv1D)                (None, 2250, 16)     400
up_sampling1d_1 (UpSampling1D)   (None, 4500, 16)     0
conv1d_4 (Conv1D)                (None, 4500, 1)      49
=================================================================
Total params: 1,105
```

### LSTM Autoencoder
An **LSTM-based** model that captures sequential dependencies:
- Uses **stacked LSTM layers** for encoding and decoding.
- Suitable for **complex time-series denoising**.
- Uses **Mean Squared Error (MSE)** as the loss function.

### Enhanced Conv1D Autoencoder
- **Deeper architecture** with additional convolutional layers.
- Improved feature extraction and reconstruction.
- Better at handling **high-frequency noise**.

## Results
- Each model is evaluated by plotting the **original, noisy, and denoised signals**.
- Performance is visually compared across models.

### Model Performance Summary:
| Model | Best Use Case |
|--------|---------------------------|
| Conv1D Autoencoder | Moderate noise levels, balanced accuracy |
| LSTM Autoencoder | Capturing long-term dependencies in signals |
| Enhanced Conv1D Autoencoder | Best overall performance, deeper architecture |

## Conclusion
This project demonstrates the effectiveness of **deep learning-based Noise Reduction techniques** for sine wave signals. Future work includes:
- Experimenting with different **model architectures**.
- Adjusting **window sizes**.
- Incorporating **advanced filtering techniques** for better noise suppression.
