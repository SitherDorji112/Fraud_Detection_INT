# fraud_app/views.py
import os
import numpy as np
from django.shortcuts import render
from django.conf import settings
from tensorflow.keras.models import load_model

# Load the ANN model once
model_path = os.path.join(settings.BASE_DIR, 'predictor', 'model', 'Fraud_Detection_model.h5')
model = load_model(model_path)


def index(request):
    prediction = None


    if request.method == 'POST':
        v11 = request.POST.get('v11', '').strip()
        v4 = request.POST.get('v4', '').strip()
        v2 = request.POST.get('v2', '').strip()
        v19 = request.POST.get('v19', '').strip()
        v8 = request.POST.get('v8', '').strip()

        if all([v11, v4, v2, v19, v8]):
            try:
                v11 = float(v11)
                v4 = float(v4)
                v2 = float(v2)
                v19 = float(v19)
                v8 = float(v8)

                input_data = np.array([[v11, v4, v2, v19, v8]])
                result = model.predict(input_data)
                prediction = 'Fraud' if result[0][0] > 0.5 else 'Not Fraud'
            except ValueError:
                prediction = 'Invalid input. Please enter valid numbers.'
        else:
            prediction = 'Please fill in all fields.'


    return render(request, 'predictor/index.html', {'prediction': prediction})

