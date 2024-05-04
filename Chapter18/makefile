serve-local:
	uvicorn server:app --reload

sample-predict:
	wget https://i.ibb.co/Hzd6TxT/image.jpg -O /tmp/non-defect.png
	echo "Saved image at /tmp/non-defect.png"
	curl -X POST "http://127.0.0.1:8000/predict" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@/tmp/non-defect.png;type=image/png"

docker-build:
	docker build -t sdd:latest .

docker-run:
	docker run -p 5000:5000 sdd:latest

sample-docker-predict:
	wget https://i.ibb.co/Hzd6TxT/image.jpg -O /tmp/non-defect.png
	echo "Saved image at /tmp/non-defect.png"
	curl -X POST "http://127.0.0.1:5000/predict" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@/tmp/non-defect.png;type=image/png"
