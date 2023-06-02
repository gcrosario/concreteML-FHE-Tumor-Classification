import tkinter as tk
import customtkinter

from tkinter import messagebox, filedialog

from customtkinter import (
    CTk,
    CTkButton,
    CTkEntry,
    CTkFont,
    CTkFrame,
    CTkLabel,
    IntVar,
    StringVar,
    set_appearance_mode,
    set_default_color_theme)

import os, requests, stat, pandas
from pandas import DataFrame, read_csv

class ClientGUI:
    def __init__(self):

        # Building the user interface of the app

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

        # Feature Selection
        self.feat_selection_frame = CTkFrame(self.root)
        self.feat_selection_label = CTkLabel(self.feat_selection_frame)
        self.feat_selection_label.configure(
            text='Upload your .csv file for feature selection:',
            justify='center',
        )
        self.feat_selection_label.grid(row=0, column=0, padx=10, pady=10, sticky="nw")
        
        self.feat_selection_var = StringVar()
        self.feat_selection_filename = CTkEntry(self.feat_selection_frame, textvariable=self.feat_selection_var)
        self.feat_selection_filename.configure(
            placeholder_text = 'input filepath or browse...',
            placeholder_text_color = 'grey',
            justify='center',
            width=500,
            exportselection=False,
            # state="disabled",
            takefocus=False,
        )
        self.feat_selection_filename.grid(row=1, column=0, padx=10)
        # self.feat_selection_chosen_file = CTkLabel(self.feat_selection_filename)
        # self.feat_selection_filename.pack()

        self.feat_selection_browse = CTkButton(
            self.feat_selection_frame, 
            hover=True,
            # command=
            )
        self.feat_selection_browse.configure(
            hover_color='#2c780b', 
            text='Browse File', 
            command=self.browseFiles
            )
        self.feat_selection_browse.grid(row=1, column=2, padx=10)

        self.feat_selection = CTkButton(
            self.feat_selection_frame, 
            command=self.dropColumns
            )
        self.feat_selection.configure(
            hover_color="#2c780b",
            text='Submit and Begin Feature Selection',
            width=300
            )
        self.feat_selection.grid(row=2, column=0, columnspan=3, pady=10)

        self.feat_selection_frame.pack(
            anchor="w",
            fill="x",
            padx=20,
            pady=10,
            side="top"
            )
        
        # Encryption
        self.encryption_frame = CTkFrame(self.root)
        self.encryption_label = CTkLabel(self.encryption_frame)
        self.encryption_label.configure(
            text='Upload your feature selection output file for encryption:',
            justify='center',
            )
        self.encryption_label.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

        self.encryption_var = StringVar()
        self.encryption_filename = CTkEntry(self.encryption_frame, textvariable=self.encryption_var)
        self.encryption_filename.configure(
            placeholder_text = 'input filepath or browse...',
            placeholder_text_color = 'grey',
            justify='center',
            width=500,
            exportselection=False,
            state="disabled",
            takefocus=False,
            )
        self.encryption_filename.grid(row=1, column=0, padx=10)

        self.encryption_browse = CTkButton(
            self.encryption_frame, 
            hover=True, 
            # command=self.getEncryptInput
            )
        self.encryption_browse.configure(
            hover_color='#2c780b', 
            text='Browse File', 
            command=self.browseFiles
        )
        self.encryption_browse.grid(row=1, column=2, padx=10)
        
        self.encryption = CTkButton(self.encryption_frame)
        self.encryption.configure(
            hover_color="#2c780b",
            text='Encrypt Data',
            width=300
            )
        self.encryption.grid(column=0, columnspan=3, pady=10, row=2)

        self.encryption_frame.pack(
            anchor="w",
            fill="x",
            padx=20,
            pady=10,
            side="top"
            )
        
        # Decryption
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
            # command=self.getEncryptInput
            )
        self.decryption_browse.configure(
            hover_color='#2c780b', 
            text='Browse File', 
            command=self.browseFiles
        )
        self.decryption_browse.grid(row=1, column=2, padx=10)
        
        self.decryption = CTkButton(self.decryption_frame)
        self.decryption.configure(
            hover_color="#2c780b",
            text='Decrypt Prediction',
            width=300
            )
        self.decryption.grid(column=0, columnspan=3, pady=10, row=2)

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
    
    # Browse files
    def browseFiles(self):
        filename = filedialog.askopenfilename(initialdir = "./",
                                          title = "Select a File",
                                          filetypes = (("Text files",
                                                        "*.txt*"),
                                                       ("all files",
                                                        "*.*")))
    
    # def submitRawFile(self):
    #     self.feat_selection_chosen_file.config(text=self.feature_selection_var.get())

    
    # Feature Selection
    def dropColumns(self, file, feature_names = "./kbest-top-features.txt"):
        with open(feature_names, "r") as feature_file:
            temp_list = list(f for f in feature_file.read().splitlines())

            feature_list = ['samples'] + temp_list

            drop_df = read_csv(file)
            drop_df = drop_df[[column for column in feature_list]] 
            drop_df.to_csv("./output.csv", index=False, header=True)
    
    # Get required files for client side: client.zip from the compiled fhe ML model.
    # def getClientZip():
    #     files = [
    #         r"",
    #         r"",
    #         ]
    #     for file in files:
    #         print(file.split("/")[-1].replace("%20", " "))
    #         if file.split("/")[-1].replace("%20", " ") not in os.listdir(os.path.dirname(__file__)):
    #             download(file, os.path.dirname(__file__))
        
    #     def download(url, dest_folder):
    #         if not os.path.exists(dest_folder):
    #             os.makedirs(dest_folder)

    #         filename = url.split('/')[-1].replace(" ", "_")
    #         file_path = os.path.join(dest_folder, filename)

    #         r = requests.get(url, stream=True)

    #         if r.ok:
    #             print("saving to", os.path.abspath(file_path))
    #             with open(file_path, 'wb') as f:
    #                 for chunk in r.iter_content(chunk_size=1024 * 8):
    #                     if chunk:
    #                         f.write(chunk)
    #                         f.flush()
    #                         os.fsync(f.fileno())
    #         else:
    #             print("Download failed: status code {}\n{}".format(r.status_code, r.text))

if __name__ == "__main__":

    app = ClientGUI()
    app.run()


