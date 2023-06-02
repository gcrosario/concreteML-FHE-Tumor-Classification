from django.shortcuts import render, HttpResponse, HttpResponseRedirect
from django.http import HttpResponse, FileResponse

from web_project.settings import BASE_DIR

import os, io, zipfile, requests, shutil, subprocess

from pandas import DataFrame as pd
from pandas import read_csv

from concrete.ml.deployment import FHEModelServer

def home(request):
    return render(request, 'index.html', 
                  context = {'classes_list':{0: 'ependymoma', 1: 'glioblastoma', 2: 'medulloblastoma', 3: 'normal', 4: 'pilocytic_astrocytoma'}}
                  )

def start_classification(request):

    clean_predictions_folder()

    count = 0
    model_path =os.path.join(BASE_DIR, "FHE-Compiled-Model/LR-Kbest20-Trial3/") 
    keys_path = os.path.join(BASE_DIR, "tumorClassifier/keys")
    keys_file = request.FILES['keys_file']
    pred_dir = os.path.join(BASE_DIR, "tumorClassifier/predictions")

    data = request.FILES['inputs'].read().split(b'\n\n\n\n\n')
    #del request.session['data_dict']

    enc_file_list = []

    # print([type(d) for d in data])

    #print(data)
    #print(keys_file.read())

    #print(data)

    for encrypted_input in data:
        print(encrypted_input[:200])
        count += 1
        serialized_evaluation_keys = keys_file.read()
        encrypted_prediction = FHEModelServer(model_path).run(encrypted_input, serialized_evaluation_keys)
        pred_file_name = f"encrypted_prediction_{count}.enc"
        pred_file_path = os.path.join(pred_dir, pred_file_name)
        with open(pred_file_path, "wb") as f:
            f.write(encrypted_prediction)

        #send all predictions as a zip file to client
        enc_file_list.append(pred_file_path)

    zipfile = create_zip(enc_file_list)

    return zipfile

def create_zip(file_list):
    count = 0
    zip_filename = os.path.join(BASE_DIR, "tumorClassifier/predictions/enc_predictions.zip")
    zip_download_name = "enc_predictions.zip"
    buffer = io.BytesIO()
    zip_file = zipfile.ZipFile(buffer, 'w')
    #zip_file = zipfile.ZipFile(zip_filename, 'w')
    
    for filename in file_list:
        count += 1
        with open(filename, "rb") as file_read:
            zip_file.write(filename, f"encrypted_prediction_{count}.enc")
    zip_file.close()

    #craft download response    
    resp = HttpResponse(buffer.getvalue(), content_type = "application/force-download")
    resp['Content-Disposition'] = f'attachment; filename={zip_download_name}'

    return resp

def clean_predictions_folder():
    pred_dir = os.path.join(BASE_DIR, f"tumorClassifier/predictions")

    if(os.listdir(pred_dir)):
        for f in os.listdir(pred_dir):
            os.remove(os.path.join(pred_dir, f))
