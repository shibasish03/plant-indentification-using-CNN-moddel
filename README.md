# 🌿 Plant Detection Using Machine Learning and Image Processing

This project uses a custom-trained Convolutional Neural Network (CNN) to classify plant types. The model is trained using a labeled dataset from Kaggle and deployed through a user-friendly Streamlit interface.

## 📁 Project Structure

├── train.py # Training script for the CNN model
├── ui.py # Streamlit-based user interface for prediction
├── model/ # Directory to store trained model (.pt)
├── dataset/ # Directory to place training and testing images
├── requirements.txt
└── README.md



## 📦 Dataset

The dataset used for training the plant classification model can be downloaded from Kaggle.

🔗 **Download Dataset**: [Plant Type Dataset on Kaggle](https://www.kaggle.com/datasets/yudhaislamisulistya/plants-type-datasets)

After downloading, extract and place the dataset contents inside the `dataset/` directory like so:



> **Note:** Make sure the folder structure is preserved as it is used for labeling during training.

---

## 🧠 Training the Model

Use train.py to train the CNN model on the dataset.

python train.py
This will:

Load and preprocess the dataset

Train a CNN model

Save the trained model to model/plant_classifier.pt

🖥️ Running the Streamlit UI
Use ui.py to launch the interactive web interface for prediction.

bash
Copy
Edit
streamlit run ui.py
With this interface, you can:

Upload a plant image

View the predicted plant type

See class probabilities and related information (if implemented)

📌 Requirements
Install required dependencies with:

bash
Copy
Edit
pip install -r requirements.txt
Typical dependencies include:

torch

torchvision

streamlit

Pillow

numpy

✅ Features
Custom CNN model for plant classification

Real-time image upload and prediction via web UI

Easy-to-train and deploy architecture

Based on open-source dataset

📬 Contact
For any queries or feedback, feel free to open an issue or submit a pull request.

⭐ If you find this useful, consider giving a star to the repo!
