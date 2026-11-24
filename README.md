# Electronic Object Classifier

Upload one or more JPG or PNG images. The app displays the original image, original-resolution Canny edge map without display dilation, a ConvNeXt first-convolution activation overlay, and an ensemble confidence score. Predictions use soft voting across the notebook-trained seed-42, seed-100, and seed-123 models. The displayed 94.40% is the mean of their individual held-out test accuracies (143 images per seed), not an ensemble accuracy measurement. Model inference still uses the notebook's 224×224 Canny preprocessing. Predicted class names remain hidden.

## Run locally

From this folder in Command Prompt:

```cmd
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

All three trained checkpoints are included in this project folder. The seed-123 checkpoint is `dual_stream_model.keras`; the seed-42 and seed-100 checkpoints are under `models/seed_42/` and `models/seed_100/`. No model upload is required.
