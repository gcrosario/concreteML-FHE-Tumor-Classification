import tkinter as tk
import customtkinter

from tkinter import messagebox, filedialog, END, INSERT

from customtkinter import (
    CTk,
    CTkButton,
    CTkEntry,
    CTkFont,
    CTkFrame,
    CTkLabel,
    CTkTextbox,
    IntVar,
    StringVar,
    set_appearance_mode,
    set_default_color_theme)

import os, requests, stat, pathlib, shutil, subprocess, zipfile, traceback, urllib
import pandas, numpy

from pandas import DataFrame, read_csv
from numpy import save

from concrete.ml.deployment import FHEModelClient

class ClientGUI:
    def __init__(self, master=None):

        # Initialize FHEModelClient
        self.fhe_model_client = FHEModelClient(os.path.dirname(__file__), os.path.join(os.path.dirname(__file__), "keys"))
        self.data_dictionary = {}

        # create required folders if not exists
        this_folder = os.path.dirname(__file__)

        required_folder_names = ["keys", "predictions"]

        for name in required_folder_names:
            if not os.path.exists(os.path.join(this_folder, f"{name}")):
                os.mkdir(os.path.join(this_folder, f"{name}"))

        ### Building the user interface of the app

        # Initialize customtkinter
        self.root = CTk()

        # System Settings
        set_appearance_mode("system")
        set_default_color_theme("green")

        # Custom CTk appearance
        self.root.configure(padx=20, pady=20)
        self.root.geometry("800x550")
        self.root.resizable(True, True)
        self.root.title("FHE-based Brain Tumor Classifier")

        # Top Label
        self.title = CTkLabel(self.root)
        self.title.configure(
            text='FHE-based Brain Tumor Classifier', 
            fg_color='#b4ff99',
            font=CTkFont(family='Arial', size=25, weight='bold'),
            text_color='#1d2b17',
            justify ='center'
            )
        self.title.pack(fill='x')

        # Description of the App

        ### Data Preprocessing: Feature Selection and Encryption

        self.preprocessing_frame = CTkFrame(self.root)
        self.preprocessing_label = CTkLabel(self.preprocessing_frame)
        self.preprocessing_label.configure(
            text='Upload your .csv file for feature selection and encryption:',
            justify='center',
            )
        self.preprocessing_label.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

        self.preprocessing_var = StringVar()
        self.preprocessing_filename = CTkEntry(self.preprocessing_frame, textvariable=self.preprocessing_var)
        self.preprocessing_filename.configure(
            placeholder_text = 'input filepath or browse...',
            placeholder_text_color = 'grey',
            justify='center',
            width=500,
            exportselection=False,
            state="disabled",
            takefocus=False,
            )
        self.preprocessing_filename.grid(row=1, column=0, padx=10)

        self.preprocessing_browse = CTkButton(
            self.preprocessing_frame, 
            hover=True,
            )
        self.preprocessing_browse.configure(
            hover_color='#2c780b', 
            text='Browse File', 
            command=self.browseRawFile
        )
        self.preprocessing_browse.grid(row=1, column=2, padx=10)
        
        self.preprocessing_begin = CTkButton(self.preprocessing_frame)
        self.preprocessing_begin.configure(
            hover_color="#2c780b",
            text='Preprocess Data',
            width=300,
            command = self.preprocessInput
            )
        self.preprocessing_begin.grid(row=2, column=0, columnspan=3, pady=10)

        preprocessing_output_frame = CTkFrame(self.root)
        self.preprocessing_output_label = CTkLabel(preprocessing_output_frame)
        self.preprocessing_output_label.configure(text='Output Window')
        self.preprocessing_output_label.pack(side="top")
        self.preprocessing_output = CTkTextbox(preprocessing_output_frame)
        self.preprocessing_output.configure(height=75, state="disabled")
        _text_ = 'Track the status of your data here.'
        self.preprocessing_output.configure(state="normal")
        self.preprocessing_output.insert("0.0", _text_)
        self.preprocessing_output.configure(state="disabled")
        self.preprocessing_output.pack(expand=True, fill="both", padx=10, pady=10)
        preprocessing_output_frame.pack(expand=True, fill="both", padx=20, pady=10, side="top")

        # self.preprocessing_output_fs = CTkTextbox(self.preprocessing_frame)
        # self.preprocessing_output_fs.configure(height=25, state="disabled", width=600)
        # _text_ = 'The first 100 bits of your encryption output will be displayed here.'
        # self.preprocessing_output_fs.insert("0.0", _text_)
        # self.encryption_output.configure(
        #     state="disabled",
        #     )
        # self.preprocessing_output_fs.grid(row=3, column=0, columnspan=3, pady=10, sticky="s")

        # self.preprocessing_output_enc = CTkTextbox(self.preprocessing_frame)
        # self.preprocessing_output_enc.configure(height=25, state="disabled", width=600)
        # _text_ = 'The first 100 bits of your encryption output will be displayed here.'
        # self.preprocessing_output_fs.insert("0.0", _text_)
        # self.encryption_output.configure(
        #     state="disabled",
        #     )
        # self.preprocessing_output_enc.grid(row=4, column=0, columnspan=3, pady=10, sticky="s")

        self.preprocessing_frame.pack(
            anchor="w",
            fill="x",
            padx=20,
            pady=10,
            side="top"
            )
        
        ### Decryption
        self.decryption_frame = CTkFrame(self.root)
        self.decryption_label = CTkLabel(self.decryption_frame)
        self.decryption_label.configure(
            text='Upload your FHE prediction output file for decryption:',
            justify='center',
            )
        self.decryption_label.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

        self.decryption_var = StringVar()
        self.decryption_filename = CTkEntry(self.decryption_frame, textvariable=self.decryption_var)
        self.decryption_filename.configure(
            placeholder_text = 'input filepath or browse...',
            placeholder_text_color = 'grey',
            justify='center',
            width=500,
            exportselection=False,
            state="disabled",
            takefocus=False,
            )
        self.decryption_filename.grid(row=1, column=0, padx=10)

        self.decryption_browse = CTkButton(
            self.decryption_frame, 
            hover=True, 
            )
        self.decryption_browse.configure(
            hover_color='#2c780b', 
            text='Browse File', 
            command=self.browsePrediction
        )
        self.decryption_browse.grid(row=1, column=2, padx=10)
        
        self.decryption_begin = CTkButton(self.decryption_frame)
        self.decryption_begin.configure(
            hover_color="#2c780b",
            text='Decrypt Prediction',
            width=300,
            command = self.decryptPrediction
            )
        self.decryption_begin.grid(column=0, columnspan=3, pady=10, row=2)

        self.decryption_frame.pack(
            anchor="w",
            fill="x",
            padx=20,
            pady=10,
            side="top"
            )

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Main Widget
        self.mainwindow = self.root

    def run(self):
        self.mainwindow.mainloop()

    def on_closing(self):
        if messagebox.askyesno(title="Quit?", message="Do you really want to quit?"):
            self.root.destroy()

    def writeOutput(self, string, delete_switch = False):
        """Function for writing argument 'string' to the app's output window. Set argument 'delete_switch' to True to clear the window before printing."""
        self.preprocessing_output.configure(state="normal")
        if(delete_switch):
            self.preprocessing_output.delete("1.0", END) #tk.END
            self.preprocessing_output.insert("0.0", f"{string}\n\n")
        else:
            self.preprocessing_output.insert(INSERT, f"{string}\n\n")
        self.preprocessing_output.see(END)
        self.preprocessing_output.configure(state="disabled")
    
    ### Preprocessing with feature selection and encryption combined
    def preprocessInput(self):
        self.dropColumns()
        self.encryptInput()

    ## Feature Selection
    def browseRawFile(self):
        filename = filedialog.askopenfilename(initialdir = "./",
                                          title = "Select a File",
                                        #   filetypes = (("all files","*.*"))
                                          )
        # self.feat_selection_var.set(filename)
        self.preprocessing_var.set(filename)
    
    def dropColumns(self, feature_names = "./client-gui/kbest-top-features.txt"):
        # filename = self.feat_selection_var.get()
        filename = self.preprocessing_var.get()

        if(not filename.endswith(".csv")):
                raise Exception("Invalid file type. Only .csv files are supported.")

        self.writeOutput("Beginning to process your data for feature selection...")
        
        with open(feature_names, "r") as feature_file:
            temp_list = list(f for f in feature_file.read().splitlines())

            feature_list = ['samples'] + temp_list

            drop_df = read_csv(filename)
            drop_df = drop_df[[column for column in feature_list]] 
            drop_df.to_csv("./client-gui/feature_selection_output.csv", index=False, header=True)
        
        self.writeOutput("Feature Selection DONE!")

        # self.preprocessing_output_fs.configure(state="normal")
        # self.preprocessing_output_fs.delete("1.0", END)
        # self.preprocessing_output_fs.insert("0.0", text="Feature Selection DONE!")
        # self.preprocessing_output_fs.configure(state="disabled")
        
    
    def generateKeys(self):
        model_dir = os.path.dirname(__file__)
        key_dir = os.path.join(os.path.dirname(__file__), "keys")

        if(os.listdir(key_dir)):
            for f in os.listdir(key_dir):
                shutil.rmtree(os.path.join(key_dir, f))

        fhemodel_client = FHEModelClient(model_dir, key_dir=key_dir)

        # The client first need to create the private and evaluation keys.
        fhemodel_client.generate_private_and_evaluation_keys()

        # Get the serialized evaluation keys
        self.serialized_evaluation_keys = fhemodel_client.get_serialized_evaluation_keys()

    def saveEncryption(self):
        filename = "encrypted_input.txt"
        with open(os.path.join(os.path.dirname(__file__), filename), "wb") as enc_file:
            for line in self.encrypted_rows:
                enc_file.write(line)
                # enc_file.write(b"\n\n\n\n\n")
        
        with open(os.path.join(os.path.dirname(__file__), r'serialized_evaluation_keys.ekl'), "wb") as f:
            f.write(self.serialized_evaluation_keys)

    def sendEncryptRequestToServer(self, client):
        """Sends 'encrypted_input.txt' and 'serialized_evaluation_keys.ekl' (expected to be located in the same directory as the app) to the server-side app through the Python requests library. URL is currently set to localhost:8000 for development purposes."""
        
        self.writeOutput("Sending encrypted input and keys to server for classification...")
        
        app_url = "http://localhost:8000"

        if 'csrftoken' in client.cookies:
            # Django 1.6 and up
            csrftoken = client.cookies['csrftoken']
        else:
            # older versions
            csrftoken = client.cookies['csrf']

        #self.writeOutput(f"{type(self.encrypted_rows)}")

        eval_keys_file = open(
                            (os.path.join(os.path.dirname(__file__), "serialized_evaluation_keys.ekl")),
                            "rb"
                            )
        inputs_file = open(
                        (os.path.join(os.path.dirname(__file__), "encrypted_input.txt")),
                        "rb"
                        )
        
        request_data = dict(csrfmiddlewaretoken=csrftoken)
        request_files = dict(inputs=inputs_file, keys_file=eval_keys_file)
        
        #code to send the above files to "localhost:8000/{function_name}"
        request_output = client.post(f"{app_url}/start_classification", data = request_data, files=request_files, headers=dict(Referer=app_url))

        if("test.zip" in os.listdir(os.path.dirname(__file__))): os.remove(os.path.join(os.path.dirname(__file__), "test.zip"))

        with open(os.path.join(os.path.dirname(__file__), "predictions/enc_predictions.zip"), "wb") as z:
            z.write(request_output.content)

        self.writeOutput("Sending completed!")

    def encryptInput(self):
        try: 

            for f in os.path.join(os.path.dirname(__file__)):
                if f.split("/")[-1] in ["encrypted_input.txt", "serialized_evaluation_keys.ekl"]:
                    os.remove(f)

            self.writeOutput("Generating Keys...")

            self.generateKeys()

            encryption_input = os.path.join(os.path.dirname(__file__), "feature_selection_output.csv")
            df = read_csv(encryption_input)
            arr_no_id = df.drop(columns=['samples']).to_numpy(dtype="uint16")

            # encrypted rows for input to server
            encrypted_rows = []

            #encrypted dictionary for outputs
            count = 0
            for id in df['samples']:
                self.data_dictionary[count] = {'id':id, 'result':''} 

            for row in range(0, arr_no_id.shape[0]):
                #clear_input = arr[:,1:]
                clear_input = arr_no_id[[row],:]
                #print(clear_input)
                encrypted_input = self.fhe_model_client.quantize_encrypt_serialize(clear_input)
                self.writeOutput(f"New row encrypted of {type(encrypted_input)}; adding to list of encrypted values...")
                encrypted_rows.append(encrypted_input)
            
            self.encrypted_rows = encrypted_rows
            
            for row in encrypted_rows:
                print("Row: ", row[:10])

            self.writeOutput(f"Encryption complete! The first 15 character of your encrypted output:\n{encrypted_rows[0][0:16]}")

            self.saveEncryption()

            self.writeOutput("Encrypted inputs and key files saved to 'encrypted_input.txt' and 'serialized_evaluation_keys.ekl'. Please do not move these files until after prediction.")

            app_url = "http://localhost:8000"

            client = requests.session()

            client.get(app_url)

            self.sendEncryptRequestToServer(client=client)

            # self.preprocessing_output_enc.configure(state="normal")
            # # self.preprocessing_output.delete("1.0", END) #tk.END
            # # self.preprocessing_output.insert("0.0", f"Your encrypted output:\n{encrypted_rows[0][0:16]}")
            # self.preprocessing_output_enc.insert(END, text="Encryption DONE!")
            # self.preprocessing_output_enc.configure(state="disabled")

        except Exception as e:
            self.writeOutput(f"Error: {traceback.format_exc()}")
    
    ### Decryption
    def browsePrediction(self):
        filename = filedialog.askopenfilename(initialdir = "./",
                                          title = "Select a File",
                                        #   filetypes = (("all files","*.*"))
                                          )
        self.decryption_var.set(filename)
    
    def decryptPrediction(self):
        try: 
            filename = self.decryption_var.get()

            if not filename.endswith(".zip"):
                raise Exception("Invalid file type: Only .zip files are supported.")

            decrypted_predictions = []
            classes_dict = {0: 'ependymoma', 1: 'glioblastoma', 2: 'medulloblastoma', 3: 'normal', 4: 'pilocytic_astrocytoma'}
            pred_folder = os.path.join(os.path.dirname(__file__), "predictions")

            zip_name = filename #os.path.join(pred_folder, "enc_predictions.zip") if os.listdir(pred_folder) else os.path.join(os.path.dirname(__file__), "enc_predictions.zip")

            with zipfile.ZipFile(zip_name, "r") as zObject:
                zObject.extractall(path=pred_folder)
            
            enc_file_list = [filename for filename in os.listdir(pred_folder) if filename.endswith(".enc")]

            for filename in enc_file_list:
                print(filename)
                with open(os.path.join(pred_folder, filename), "rb") as f:
                    decrypted_prediction = self.fhe_model_client.deserialize_decrypt_dequantize(f.read())[0]
                    decrypted_predictions.append(decrypted_prediction)
            
            decrypted_predictions_classes = numpy.array(decrypted_predictions).argmax(axis=1)
            final_output = [classes_dict[output] for output in decrypted_predictions_classes]
            
            print(final_output)

            for i in range(len(final_output)):
                self.data_dictionary[i]['result'] = final_output[i]
            
            self.writeOutput([dictionary for dictionary in self.data_dictionary.values()])
        
        except Exception as e:
            self.writeOutput(f"Error: {str(e)}")

def getClientFiles():
    files = [
        r"https://github.com/gcrosario/concreteML-FHE-Tumor-Classification/raw/master/FHE-Compiled-Model/LR-Kbest20-Trial3/client.zip",
        ]
    for file in files:
        print(file.split("/")[-1].replace("%20", " "))
        if file.split("/")[-1].replace("%20", " ") not in os.listdir(os.path.dirname(__file__)):
            download(file, os.path.dirname(__file__))

def download(url, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    filename = url.split('/')[-1].replace(" ", "_")
    file_path = os.path.join(dest_folder, filename)

    r = requests.get(url, stream=True)

    if r.ok:
        print("saving to", os.path.abspath(file_path))
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
    else:  # HTTP status code 4XX/5XX
        print("Download failed: status code {}\n{}".format(r.status_code, r.text))

if __name__ == "__main__":

    download_files = input("Would you like to download the required client files? (Type Yes or No.) ")
    if(download_files.strip() in ["y", "yes", "YES", "Yes"]):
        getClientFiles()

    app = ClientGUI()
    app.run()


