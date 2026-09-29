# 🦴 MedRay AI — Bone Fracture Detection

A production-style Streamlit application for binary bone-fracture screening from X-ray images using a trained TensorFlow model.

**Stack:** Python • TensorFlow/Keras • Streamlit • Docker • GitHub Actions • Render / Hugging Face Spaces

> ⚠️ **Medical disclaimer:** This is an educational/research project and not a medical device. Predictions must not be used as a diagnosis or treatment decision.

## What changed from the original project

The original repository was primarily a notebook + Streamlit demo. This version adds a deployable application structure:

- Clean inference module with deterministic model loading.
- No random prediction fallback when the model is missing.
- Four-page Streamlit interface: Dashboard, Fracture Detection, Reports, and Case History.
- Case details captured with each analysis: case ID, age, and sex.
- JPG, JPEG, and PNG upload with input validation and explicit inference errors.
- Prediction result, model confidence, raw score, and class-probability chart.
- Grad-CAM attention overlay, displayed beside the X-ray and available as a PNG download.
- Local CSV-backed case history, searchable by case ID, with report download.
- Docker image with non-root runtime user and health check.
- Automated unit tests.
- GitHub Actions CI: tests + Docker build.
- Optional GitHub Actions → Hugging Face Space sync.
- Render Blueprint for one-click deployment.
- `MODEL_PATH` and `DATA_DIR` environment variables for model and history locations.

The existing model's inference contract is retained: **128×128 RGB input, normalized to [0,1], binary score, threshold 0.5**. The app currently maps scores below 0.5 to “Fracture Detected”; verify that label mapping against your training setup before interpreting results.

The app saves case details and prediction results to `data/history.csv` by default. Uploaded X-ray images are not added to that history file. Use only synthetic or properly de-identified information; this demo is not designed for storing protected health information. Local disk persistence depends on the deployment platform and may not survive redeployments.

## Project structure

```text
bone-fracture-detection/
├── app.py
├── check_model.py
├── src/
│   ├── __init__.py
│   ├── inference.py
│   └── storage.py
├── tests/
│   └── test_inference.py
├── dataset/
│   ├── train/{fractured,not fractured}/
│   ├── val/{fractured,not fractured}/
│   └── test/{fractured,not fractured}/
├── data/
│   └── history.csv   # created after the first analysis; ignored by Git
├── .github/workflows/
│   ├── ci.yml
│   └── deploy-huggingface.yml
├── .streamlit/config.toml
├── Dockerfile
├── .dockerignore
├── .gitignore
├── render.yaml
├── requirements.txt
├── runtime.txt
├── fracture_model.h5
└── README.md
```

## 1. Model and data paths

The repository includes the trained model at `fracture_model.h5`, which is the default model path. To use a different model file, set the `MODEL_PATH` environment variable. Set `DATA_DIR` to change where the app writes `history.csv`; by default, it uses the project-level `data/` directory.

The model loads when an analysis is submitted. If it is missing or cannot be loaded, the app reports an inference error rather than generating a fallback prediction.

For a public GitHub repository, consider Git LFS or an external model artifact store if the binary becomes too large for normal Git workflows.

## 2. Run locally

Recommended Python: 3.11.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`.

### Run tests

```bash
pytest -q
```

## 3. Run with Docker

Make sure `fracture_model.h5` is present, then:

```bash
docker build -t bone-fracture-detection .
docker run --rm -p 8501:8501 bone-fracture-detection
```

Open `http://localhost:8501`.

Health endpoint:

```text
http://localhost:8501/_stcore/health
```

## 4. Deploy on Render

### Dashboard method

1. Push this project to GitHub.
2. Open Render and choose **New → Web Service**.
3. Connect the GitHub repository.
4. Select **Docker** as the runtime.
5. Use the `render.yaml` settings or set the health check path to `/_stcore/health`.
6. Deploy.

The included Blueprint configures a Docker web service and enables automatic deploys from the connected repository.

### Important: model file

The deployment must have access to `fracture_model.h5`. If you keep the model in the repository, Docker will copy it into the image. If you move the model to external storage later, change `MODEL_PATH` and add the download step during deployment.

## 5. Hugging Face Spaces option

Hugging Face Spaces supports Docker-based applications. Create a Docker Space, then configure the included workflow to sync this repository to it.

Set:

- GitHub repository variable: `HF_SPACE` = `YOUR_USERNAME/YOUR_SPACE`
- GitHub secret: `HF_TOKEN` = a Hugging Face token with write access

## 6. GitHub Actions

Every push/PR to `main` runs:

1. Python dependency installation
2. Unit tests
3. Docker image build

This gives you a visible CI pipeline under the repository's **Actions** tab.

## Recruiter demo flow

Use this 60-second demo:

1. Open the live link.
2. Go to **Fracture Detection**.
3. Upload a sample X-ray.
4. Show the prediction, confidence, probability chart, and Grad-CAM overlay.
5. Open **Case History** to search by case ID, then **Reports** to download the CSV.
6. Mention: “The app is containerized with Docker and automatically tested/build-checked through GitHub Actions.”

## Resume bullet

**Bone Fracture Detection | TensorFlow, Streamlit, Docker, GitHub Actions** — Deployed a deep-learning X-ray classification application with a production-style Streamlit interface, deterministic model inference, case history/reporting, Docker containerization, automated tests, CI/CD, and cloud deployment configuration.

## Model / dataset

The original project uses a labeled bone X-ray dataset and compares deep-learning approaches including VGG16, ResNet and a custom CNN. The current deployment uses the trained `fracture_model.h5` artifact from the repository.
