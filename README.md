# 🦴 MedRay AI — Bone Fracture Detection

A production-style Streamlit application for binary bone-fracture screening from X-ray images using a trained TensorFlow model.

**Stack:** Python • TensorFlow/Keras • Streamlit • Docker • GitHub Actions • Render / Hugging Face Spaces

> ⚠️ **Medical disclaimer:** This is an educational/research project and not a medical device. Predictions must not be used as a diagnosis or treatment decision.

## What changed from the original project

The original repository was primarily a notebook + Streamlit demo. This version adds a deployable application structure:

- Clean inference module with deterministic model loading.
- No random prediction fallback when the model is missing.
- Persistent demo history isolated in `data/`.
- Case search and CSV report download.
- Input validation and explicit inference errors.
- Docker image with non-root runtime user and health check.
- Automated unit tests.
- GitHub Actions CI: tests + Docker build.
- Optional GitHub Actions → Hugging Face Space sync.
- Render Blueprint for one-click deployment.
- Environment variables for model/data locations.

The existing model's inference contract is retained: **128×128 RGB input, normalized to [0,1], binary score, threshold 0.5**. The current repository's app maps scores below 0.5 to “Fracture Detected”, so that mapping is preserved. Verify the label mapping against your training notebook before presenting clinical-style metrics. 

## Project structure

```text
bone-fracture-detection/
├── app.py
├── src/
│   ├── __init__.py
│   ├── inference.py
│   └── storage.py
├── tests/
│   └── test_inference.py
├── data/
│   └── .gitkeep
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
├── fracture_model.h5   # copy your existing model here
└── README.md
```

## 1. Put your trained model in the project

Your current GitHub repository already contains `fracture_model.h5`. Keep that file in the project root:

```text
fracture_model.h5
```

Do **not** retrain the model just for deployment. The deployment app loads the trained artifact at runtime.

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

Render currently provides free web services, but free instances have a **512 MB RAM** limit and spin down after 15 minutes without traffic. A TensorFlow application may require more memory than that, so if the container fails during model loading, select a larger web-service plan. citeturn3search1turn3search2

### Dashboard method

1. Push this project to GitHub.
2. Open Render and choose **New → Web Service**.
3. Connect the GitHub repository.
4. Select **Docker** as the runtime.
5. Use the `render.yaml` settings or set the health check path to:
   `/ _stcore/health` (without the space).
6. Deploy.

Render supports Docker-based web services and gives the service an `onrender.com` URL.

### Important: model file

The deployment must have access to `fracture_model.h5`. If you keep the model in the repository, Docker will copy it into the image. If you move the model to external storage later, change `MODEL_PATH` and add the download step during deployment.

## 5. Hugging Face Spaces option

Hugging Face Spaces supports Docker-based applications. For a Docker Space, the Space metadata should use `sdk: docker` and an appropriate `app_port`; Streamlit applications can therefore be packaged as Docker apps. citeturn0search3turn0search0

Create a new Docker Space and sync this repository to it. The included workflow can do the sync automatically.

Set:

- GitHub repository variable: `HF_SPACE` = `YOUR_USERNAME/YOUR_SPACE`
- GitHub secret: `HF_TOKEN` = a Hugging Face token with write access

The official `huggingface/hub-sync` action supports GitHub → Spaces synchronization. citeturn0search2turn0search4

## 6. GitHub Actions

Every push/PR to `main` runs:

1. Python dependency installation
2. Unit tests
3. Docker image build

This gives you a visible CI pipeline under the repository's **Actions** tab. GitHub documents this workflow pattern for Python CI and continuous integration. citeturn3search5turn3search12

## Recruiter demo flow

Use this 60-second demo:

1. Open the live link.
2. Go to **Fracture Detection**.
3. Upload a sample X-ray.
4. Show the prediction + confidence + probability chart.
5. Open **Reports** and download the CSV.
6. Mention: “The app is containerized with Docker and automatically tested/build-checked through GitHub Actions.”

## Resume bullet

**Bone Fracture Detection | TensorFlow, Streamlit, Docker, GitHub Actions** — Deployed a deep-learning X-ray classification application with a production-style Streamlit interface, deterministic model inference, case history/reporting, Docker containerization, automated tests, CI/CD, and cloud deployment configuration.

## Model / dataset

The original project uses a labeled bone X-ray dataset and compares deep-learning approaches including VGG16, ResNet and a custom CNN. The current deployment uses the trained `fracture_model.h5` artifact from the repository.
