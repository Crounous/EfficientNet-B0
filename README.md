# Diabetic Retinopathy Web App

This is a simple web application built with Flask that uses a pre-trained EfficientNet-B0 model to classify DR Images.

## Overview

The application provides a web interface where users can upload an image. The backend processes the image, feeds it into the trained model, and returns the predicted DR class and confidence score.

## Getting Started

Follow these instructions to get the application running locally.

### Prerequisites

* Python 3.6+
* Flask
* PyTorch
* Pillow

### Setup

1.  **Clone the Repository:**
    ```
    git clone [https://github.com/Crounous/EfficientNet-B0.git](https://github.com/Crounous/EfficientNet-B0.git)
    cd EfficientNet-B0
    ```

2.  **Install Dependencies:**
    It's highly recommended to use a virtual environment.
    ```
    # Create and activate a virtual environment
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

    # Install the required packages
    pip install flask torch torchvision Pillow
    (Or simply run "pip install -r requirements.txt")
     ```
## Running the Application

Once the setup is complete, you can start the Flask server by running the `app.py` script:

The application will be available at `http://127.0.0.1:5000` in your web browser.

## How to Use

1.  Open your web browser and navigate to `http://127.0.0.1:5000`.
2.  Click the "Choose File" button to select an image from your computer.
3.  The application will display the predicted class and the confidence level of the prediction.
