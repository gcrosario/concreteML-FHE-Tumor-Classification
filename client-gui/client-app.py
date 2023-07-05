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

import os, requests, stat, pathlib, shutil, subprocess, zipfile, traceback, urllib, json
import pandas, numpy
from pandas import DataFrame, read_csv
from numpy import save
from datetime import datetime
from concrete.ml.deployment import FHEModelClient

class ClientGUI:
    def __init__(self, master=None):

        # Initialize FHEModelClient
        self.fhe_model_client = FHEModelClient(os.path.dirname(__file__), os.path.join(os.path.dirname(__file__), "keys"))
        self.data_dictionary = {}


        # Create folder for keys and predictions if they don't exist
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
        self.root.configure(padx=20, pady=20, 
                            fg_color='#3d4539',
                            )
        self.root.geometry("900x950")
        self.root.resizable(True, True)
        self.root.title("FHE-based Brain Tumor Classifier (Client)")

        # Top Label
        self.title = CTkLabel(self.root)
        self.title.configure(
            text='FHE-based Multi-Class Brain Tumor Classifier (Client)', 
            fg_color='#aee895',
            font=CTkFont(size=30, weight='bold'),
            text_color='#1d2b17',
            justify ='center',
            )
        self.title.pack(fill='x', pady = 10,)

        # Description of the App
        self.description_frame = CTkFrame(self.root)
        self.about_label = CTkLabel(self.description_frame)
        self.about_label.configure(
            font=CTkFont(size=20, weight='bold'),
            text='About the Tool',
            text_color='#1d2b17',
            )
        self.about_label.pack(
                            # expand=False, fill="both", 
                            pady=10, side="top"
                              )
        self.description_label = CTkLabel(self.description_frame)
        self.description_label.configure(
            justify='center',
            text='This tool implements FHE-based logistic regression model for tumor classification using gene expression data.',
            font=CTkFont(size=16),
            text_color='#1d2b17',
            fg_color='white',
            )
        self.description_label.pack(expand=False, fill="x", side="top")
        self.description_frame.pack(
            fill="both", ipady=10, padx=20, pady=20, side="top")

        # Data Preprocessing: Feature Selection and Encryption
        # Variables for filenames
        self.preprocessing_var = StringVar()
        self.decryption_var = StringVar()

        # Preprocessing Frame
        self.preprocessing_frame = CTkFrame(self.root)

        self.preprocessing_label = CTkLabel(self.preprocessing_frame)
        self.preprocessing_label.configure(
            text='Upload your .csv file for feature selection, encryption, and prediction here:',
            justify='left',
            text_color='#1d2b17',
            font=CTkFont(size=16),
            )
        self.preprocessing_label.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

        self.preprocessing_filename = CTkEntry(self.preprocessing_frame, textvariable=self.preprocessing_var)
        self.preprocessing_filename.configure(
            justify='left',
            width=640,
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
            # width=300,
            font=CTkFont(size=15),
            command=self.browseRawFile
        )
        self.preprocessing_browse.grid(row=1, column=2, padx=10)
        
        self.preprocessing_begin = CTkButton(self.preprocessing_frame)
        self.preprocessing_begin.configure(
            hover_color="#2c780b",
            text='Submit Data for FHE Classification',
            width=300,
            font=CTkFont(size=15),
            command = self.processInput
            )
        self.preprocessing_begin.grid(row=2, column=0, columnspan=3, pady=10)

        self.preprocessing_frame.pack( anchor="w", fill="x", padx=20, pady=10, side="top")
        
        # Output Status Frame
        output_tracker_frame = CTkFrame(self.root)

        self.output_tracker_label = CTkLabel(output_tracker_frame)
        self.output_tracker_label.configure(
            text='Output Window',
            text_color='#1d2b17',
            font=CTkFont(size=20, weight='bold'),
            justify='center',
            )
        self.output_tracker_label.pack(pady=10, side="top")
        self.output_tracker = CTkTextbox(output_tracker_frame)
        self.output_tracker.configure(height=75, state="disabled")
        _text_ = 'Track the status of your data here.'
        self.output_tracker.configure(state="normal",
                                      text_color='#1d2b17',
                                      font=CTkFont(size=18),
                                      )
        self.output_tracker.insert("0.0", _text_)
        self.output_tracker.configure(state="disabled")
        self.output_tracker.pack(expand=True, fill="both", padx=10, pady=10)

        output_tracker_frame.pack(expand=True, fill="both", padx=20, pady=10, side="top")

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Main Widget
        self.mainwindow = self.root

    ### Running the UI
    def run(self):
        self.mainwindow.mainloop()


    ### Verifying the action of closing the app
    def on_closing(self):
        if messagebox.askyesno(title="Quit?", message="Do you really want to quit?"):
            self.root.destroy()


    ### Function for writing argument 'string' to the app's output window. Set argument 'delete_switch' to True to clear the window before printing.
    def writeOutput(self, string, delete_switch = False):
        self.output_tracker.configure(state="normal")
        if(delete_switch):
            self.output_tracker.delete("1.0", END) #tk.END
            self.output_tracker.insert("0.0", f"{string}\n\n")
        else:
            self.output_tracker.insert(INSERT, f"{string}\n\n")
        self.output_tracker.see(END)
        self.output_tracker.configure(state="disabled")
    

    # Function for getting the size of a file (i.e. private key, eval key, encrypted input)
    def get_size(self, file_path, unit='bytes'):
        file_size = os.path.getsize(file_path)
        exponents_map = {'bytes': 0, 'kb': 1, 'mb': 2, 'gb': 3}
        if unit not in exponents_map:
            raise ValueError("Must select from \
            ['bytes', 'kb', 'mb', 'gb']")
        else:
            size = file_size / 1024 ** exponents_map[unit]
            return round(size, 3)
    

    ### Function for running the entire processing of input data from feature selection to FHE inference proper
    def processInput(self):
        self.getFeaturesAndClasses()
        self.dropColumns()
        self.encryptInput()
        self.decryptPrediction()


    ### Browse raw input file of client
    def browseRawFile(self):
        filename = filedialog.askopenfilename(initialdir = "./",
                                          title = "Select a File",
                                        #   filetypes = (("all files","*.*"))
                                          )
        self.preprocessing_var.set(filename)
    

    ### Get the classes and final features used in the model from features_and_classes.txt file
    def getFeaturesAndClasses(self, file = os.path.join(os.path.dirname(__file__), "features_and_classes.txt")):
        with open(file, "r") as fc_file:
            dictionary = json.loads(fc_file.readline())
            self.selected_features = dictionary["features"]
            self.classes_labels = dictionary["classes"]
            self.classes_labels = {int(key):value for key, value in self.classes_labels.items()}


    ### Feature Selection
    def dropColumns(self):
        filename = self.preprocessing_var.get()

        if(not filename.endswith(".csv")):
                raise Exception("Invalid file type. Only .csv files are supported.")

        self.writeOutput("Beginning to process your data for feature selection...")

        features = self.selected_features
        feature_list = ["samples"] + features

        drop_df = read_csv(filename)
        drop_df = drop_df[[column for column in feature_list]]
        drop_df.to_csv(os.path.join(os.path.dirname(__file__), "feature_selection_output.csv"), index=False, header=True)
        
        self.writeOutput("Feature Selection DONE!")


    ### Encryption of pre-processed client input (feature_selection_output.csv)
    def encryptInput(self):
        try: 

            for f in os.path.join(os.path.dirname(__file__)):
                if f.split("/")[-1] in ["encrypted_input.txt", "serialized_evaluation_keys.ekl"]:
                    os.remove(f)

            self.writeOutput("Generating keys...")

            # Client generates private key and evaluation key
            self.generateKeys()

            encryption_input = os.path.join(os.path.dirname(__file__), "feature_selection_output.csv")
            df = read_csv(encryption_input)
            arr_no_id = df.drop(columns=['samples']).to_numpy(dtype="uint16")

            # Encrypted rows for input to server
            encrypted_rows = []

            # Encrypted dictionary for outputs
            count = 0
            for id in df['samples']:
                self.data_dictionary[count] = {'id':id, 'result':''} 

            for row in range(0, arr_no_id.shape[0]):
                clear_input = arr_no_id[[row],:]
                encrypted_input = self.fhe_model_client.quantize_encrypt_serialize(clear_input)
                self.writeOutput("Encrypting pre-processed data...")
                encrypted_rows.append(encrypted_input)
            
            # Final encrypted input
            self.encrypted_rows = encrypted_rows

            self.writeOutput("Data Encryption DONE!")

            # Save encrypted input into a .txt file and the eval key into a .ekl file
            self.saveEncryption()

            self.writeOutput("Encrypted inputs and key files saved to 'encrypted_input.txt' and 'serialized_evaluation_keys.ekl'. Please do not move these files until after prediction.")

            # Size of encrypted input
            encrypted_input_path = os.path.join(os.path.dirname(__file__), "encrypted_input.txt")
            encrypted_input_size = self.get_size(encrypted_input_path, 'kb')
            print("Encrypted input size (kB): ", encrypted_input_size)

            # Initialize requests object for client-server interaction
            app_url = "http://localhost:8000"
            client = requests.session()
            client.get(app_url)

            pred_zip_name = self.sendEncryptRequestToServer(client=client)

            self.decryption_var.set(pred_zip_name)

        except Exception as e:
            self.writeOutput(f"Error: {traceback.format_exc()}")


    ### Client key generation function    
    def generateKeys(self):
        model_dir = os.path.dirname(__file__)
        key_dir = os.path.join(os.path.dirname(__file__), "keys")

        if(os.listdir(key_dir)):
            for f in os.listdir(key_dir):
                shutil.rmtree(os.path.join(key_dir, f))

        fhemodel_client = FHEModelClient(model_dir, key_dir=key_dir)

        # The client first need to create the private and evaluation keys.
        fhemodel_client.generate_private_and_evaluation_keys()

        # Get the serialized evaluation key
        self.serialized_evaluation_keys = fhemodel_client.get_serialized_evaluation_keys()

        # Check the size of the private key (in kB)
        priv_key_size = self.get_size("./keys", 'kb')
        print("Private key size (kB): ", priv_key_size)
    

    ### Saving the output of encryption into a .txt file and the generated eval key into .ekl file
    def saveEncryption(self):
        filename = "encrypted_input.txt"
        with open(os.path.join(os.path.dirname(__file__), filename), "wb") as enc_file:
            for line in self.encrypted_rows:
                enc_file.write(line)
        
        with open(os.path.join(os.path.dirname(__file__), r'serialized_evaluation_keys.ekl'), "wb") as f:
            f.write(self.serialized_evaluation_keys)
        
        # Check the size of the evaluation key (in kB)
        eval_key_size = self.get_size("./serialized_evaluation_keys.ekl", 'kb')
        print("Evaluation key size (kB): ", eval_key_size)


    ### Sends encrypted_input.txt and serialized_evaluation_keys.ekl (expected to be located in the same directory as the app) to the server-side app through the Python requests library. URL is set to localhost:8000 in development.
    def sendEncryptRequestToServer(self, client):
        
        app_url = "http://localhost:8000"

        if 'csrftoken' in client.cookies:
            # Django 1.6 and up
            csrftoken = client.cookies['csrftoken']
        else:
            # Older versions
            csrftoken = client.cookies['csrf']

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
        
        self.writeOutput("Sending encrypted data and evaluation keys to server...")
        self.writeOutput("Waiting for server's response...")
        
        # Sending the files (encrypted input and eval key) to "localhost:8000/{function_name}" (server's FHE inference function)
        request_output = client.post(f"{app_url}/start_classification", data = request_data, files=request_files, headers=dict(Referer=app_url))

        if request_output.ok:
            self.writeOutput(f"Response Code: {request_output.status_code}. FHE Classification DONE!")

            # Save FHE inference result into a .zip file
            with open(os.path.join(os.path.dirname(__file__), "predictions/enc_predictions.zip"), "wb") as z:
                z.write(request_output.content)

        return os.path.join(os.path.dirname(__file__), "predictions/enc_predictions.zip")


    ### Decryption of FHE inference result received by the client from the server
    def decryptPrediction(self):

        # Expects the input filepath (self.decrypt_name_var) to be a .zip file, and raises an error if not
        try: 
            filename = self.decryption_var.get()

            if not filename.endswith(".zip"):
                raise Exception("Invalid file type: Only .zip files are supported.")

            decrypted_predictions = []

            # Setting classes dictionary to be used for final output translation since the model used label encoding in training
            try:
                classes_dict = self.classes_labels
            except:
                classes_dict = {0: 'ependymoma', 1: 'glioblastoma', 2: 'medulloblastoma', 3: 'normal', 4: 'pilocytic_astrocytoma'}
            
            
            pred_folder = os.path.join(os.path.dirname(__file__), "predictions")
            zip_name = filename

            with zipfile.ZipFile(zip_name, "r") as zObject:
                zObject.extractall(path=pred_folder)
            
            enc_file_list = [filename for filename in os.listdir(pred_folder) if filename.endswith(".enc")]

            for filename in enc_file_list:
                with open(os.path.join(pred_folder, filename), "rb") as f:
                    decrypted_prediction = self.fhe_model_client.deserialize_decrypt_dequantize(f.read())[0]
                    decrypted_predictions.append(decrypted_prediction)
            
            decrypted_predictions_classes = numpy.array(decrypted_predictions).argmax(axis=1)
            final_output = [classes_dict[output] for output in decrypted_predictions_classes]

            for i in range(len(final_output)):
                self.data_dictionary[i]['result'] = final_output[i]
            
            decrypted_pred = [dictionary for dictionary in self.data_dictionary.values()] 
            print(decrypted_pred)

            final_str = "The classification of your sample is: " + final_output[0].upper()
            self.writeOutput(final_str)

            # Save decrypted FHE inference result into a .csv file
            self.savePrediction(decrypted_pred)
        
        except Exception as e:
            self.writeOutput(f"Error: {str(e)}")
    

    ### Save final result into a .csv file
    def savePrediction(self, dictionary):
        final_pred = pandas.DataFrame.from_dict(dictionary)
        
        now = datetime.now()
        date = now.strftime("%Y_%d_%m")

        fname = "predictions/" + date + "_final_prediction_output.csv"

        final_pred.to_csv((os.path.join(os.path.dirname(__file__), fname)), 
                           index=False, header=True)

        self.writeOutput("Your final prediction output has been saved! Check the predictions folder to view it.")


### Download required client files from the project's GitHub Repository.
### By default, targets the project's GitHub repository for downloading the files, but can be set to localhost:8000 to download from the local deployment server.
def getClientFiles():
    files = [
        r"https://github.com/gcrosario/concreteML-FHE-Tumor-Classification/raw/master/FHE-Compiled-Model/client.zip",
        r"https://raw.githubusercontent.com/gcrosario/concreteML-FHE-Tumor-Classification/master/FHE-Compiled-Model/features_and_classes.txt",
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

    print("Downloading required client files.... Client app will open once downloads are finished.")
    # Download required client files upon launching of the client GUI app
    getClientFiles()
    print("Required client files saved! Launching the app... \n")

    app = ClientGUI()
    app.run()


