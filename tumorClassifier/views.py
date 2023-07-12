from django.shortcuts import render, HttpResponse, HttpResponseRedirect
from django.http import HttpResponse, FileResponse
from web_project.settings import BASE_DIR
import os, io, zipfile, requests, shutil, subprocess, hashlib
from pandas import DataFrame as pd
from pandas import read_csv
from concrete.ml.deployment import FHEModelServer

### Loading the home page
def index(request):
    return render(request, 'index.html', 
                  context = {'classes_list':{0: 'ependymoma', 1: 'glioblastoma', 2: 'medulloblastoma', 3: 'normal', 4: 'pilocytic_astrocytoma'}}
                  )

### Send required client files to client
def serve_downloadable(request):
    filename = request.GET.get('filename')

    if request.method == "GET":
        response = download_file(filename=filename)
        return response
    elif request.method == "POST":
        return HttpResponse("Please use a GET request to access this endpoint.")

def download_file(filename):
    # assume server will always have latest version of software
    download_directories = [
        os.path.join(BASE_DIR, "tumorClassifier/Client-Downloads/"),
    ]

    for dir in download_directories:
        if filename in os.listdir(dir):
            download_fname = os.path.join(dir, filename)
            response = FileResponse(open(download_fname, 'rb'))
            response['Content-Disposition'] = f'attachment; filename={filename}'
            response['hash']=hash_file(download_fname)
            print(f"Serving {filename} to client...")
            return response

### Returns the SHA-1 has of the file passed into it
def hash_file(filename):

   # make a hash object
   h = hashlib.sha1()

   # open file for reading in binary mode
   with open(filename,'rb') as file:

       # loop till the end of the file
       chunk = 0
       while chunk != b'':
           # read only 1024 bytes at a time
           chunk = file.read(1024)
           h.update(chunk)

   # return the hex representation of digest
   return h.hexdigest()

### Perform FHE inference 
def start_classification(request):

    clean_predictions_folder()

    count = 0
    model_path =os.path.join(BASE_DIR, "tumorClassifier/FHE-Compiled-Model/") 
    keys_path = os.path.join(BASE_DIR, "tumorClassifier/keys")
    keys_file = request.FILES['keys_file']
    pred_dir = os.path.join(BASE_DIR, "tumorClassifier/predictions")

    data = request.FILES['inputs'].read().strip()
    # print(f"Data received from client is {data[:200]}")

    enc_file_list = []

    count += 1
    serialized_evaluation_keys = keys_file.read()
    encrypted_prediction = FHEModelServer(model_path).run(data, serialized_evaluation_keys)
    pred_file_name = f"encrypted_prediction_{count}.enc"
    pred_file_path = os.path.join(pred_dir, pred_file_name)
    with open(pred_file_path, "wb") as f:
        f.write(encrypted_prediction)

    # Send prediction to client as a zip file  
    enc_file_list.append(pred_file_path)
    zipfile = create_zip(enc_file_list)

    return zipfile


### Creating a zip file
def create_zip(file_list):
    count = 0
    zip_filename = os.path.join(BASE_DIR, "tumorClassifier/predictions/enc_predictions.zip")
    zip_download_name = "enc_predictions.zip"
    buffer = io.BytesIO()
    zip_file = zipfile.ZipFile(buffer, 'w')
    
    for filename in file_list:
        count += 1
        with open(filename, "rb") as file_read:
            zip_file.write(filename, f"encrypted_prediction_{count}.enc")
    zip_file.close()

    # Craft download response    
    resp = HttpResponse(buffer.getvalue(), content_type = "application/force-download")
    resp['Content-Disposition'] = f'attachment; filename={zip_download_name}'

    return resp


def clean_predictions_folder():
    pred_dir = os.path.join(BASE_DIR, f"tumorClassifier/predictions")

    if(os.listdir(pred_dir)):
        for f in os.listdir(pred_dir):
            os.remove(os.path.join(pred_dir, f))
