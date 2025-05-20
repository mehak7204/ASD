# ASD Prediction Web App

This project is a web application for predicting Autism Spectrum Disorder (ASD) using machine learning. The app allows users to input relevant data and receive a prediction based on a trained model.

## Features
- User-friendly web interface for ASD prediction
- Utilizes a trained machine learning model (`model.pkl`)
- Data visualization and result display
- Built with Flask, scikit-learn, pandas, and other popular Python libraries


![image](https://github.com/user-attachments/assets/9f32da13-d9e0-4fa9-a3fe-43146b99a2ca)
![image](https://github.com/user-attachments/assets/34bcbd24-9884-445a-a3f9-3f4f3ea81f70)
![image](https://github.com/user-attachments/assets/d281acd8-334b-438e-bef7-e43f38f5e143)


## Getting Started

### Prerequisites
- Python 3.12 or higher
- pip (Python package manager)

### Setup
1. (Optional) Create and activate a virtual environment:
   ```powershell
   python -m venv aspd
   .\aspd\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

### Running the App
```powershell
python app.py
```
The app will start on `http://127.0.0.1:5000/` by default.

## Usage
- Open the web app in your browser.
- Enter the required information in the form.
- Submit to receive an ASD prediction and view results.

## Model Training
- The model is trained using `ml_model.py` and the `autism_data.csv` dataset.
- The trained model is saved as `model.pkl`.

## License
This project is for educational purposes.

## Author
mehak7204
