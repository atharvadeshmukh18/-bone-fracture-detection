install:
	pip install -r requirements.txt

test:
	pytest -q

run:
	streamlit run app.py

docker-build:
	docker build -t bone-fracture-detection .

docker-run:
	docker run --rm -p 8501:8501 bone-fracture-detection
