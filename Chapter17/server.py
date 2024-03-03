import os, io, torch
from torch_snippets import P, makedir
from sdd import SDD
from PIL import Image
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SDD(torch.load('sdd.weights.pth', map_location=device))


server_root = P('/tmp')
templates = './templates'
static = server_root/'server/static'
files = server_root/'server/files'
for fldr in [static,files]: makedir(fldr)

app = FastAPI()
app.mount("/static", StaticFiles(directory=static), name="static")
app.mount("/files", StaticFiles(directory=files), name="files")
templates = Jinja2Templates(directory=templates)

@app.get("/")
async def read_item(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post('/uploaddata/')
async def upload_file(request: Request, file:UploadFile=File(...)):
    print(request)
    content = file.file.read()
    saved_filepath = f'{files}/{file.filename}'
    with open(saved_filepath, 'wb') as f:
        f.write(content)
    output = model.predict_from_path(saved_filepath)
    payload = {
        'request': request, 
        "filename": file.filename, 
        'output': output
    }
    return templates.TemplateResponse("home.html", payload)

@app.post("/predict")
def predict(request: Request, file:UploadFile=File(...)):
    content = file.file.read()
    image = Image.open(io.BytesIO(content))
    output = model.predict_from_image(image)
    return output
