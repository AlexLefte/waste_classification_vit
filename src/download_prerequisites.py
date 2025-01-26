import os
import requests
import zipfile
import gdown


def download_file(url, dest_path):
    """
    Descarcă un fișier de la adresa URL și salvează la destinația specificată.
    """
    print(f"Downloading from Google Drive: {url}...")
    gdown.download(url=url, output=dest_path, quiet=False)
    print(f"File downloaded to {dest_path}")


def extract_zip(zip_path, extract_to):
    """
    Extrage conținutul fișierului ZIP într-un folder specific.
    """
    if not zipfile.is_zipfile(zip_path):
        raise zipfile.BadZipFile(f"The file at {zip_path} is not a valid ZIP file.")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {zip_path} to {extract_to}")


def main():
    # Setează URL-urile pentru fișierele de descărcat
    dataset_url = "https://drive.google.com/uc?id=1KSxFVPknINQ6SgEGrCqO1TxeYssn4Lcl"
    model_url = "https://drive.google.com/uc?id=14wdd_qw1cb1TBqQXBNrwxZmwdA3WWj0K"

    # Folderele și fișierele țintă
    output_folder = "dataset"  # Folder pentru extragerea datelor
    os.makedirs(output_folder, exist_ok=True)  # Creează folderul dacă nu există

    # Descarcă și extrage setul de date
    dataset_zip_path = os.path.join(output_folder, "dataset.zip")
    download_file(dataset_url, dataset_zip_path)
    extract_zip(dataset_zip_path, output_folder)
    os.remove(dataset_zip_path)  # Șterge fișierul ZIP

    # Descarcă și extrage modelul
    output_folder = "models"
    os.makedirs(output_folder, exist_ok=True) 
    model_zip_path = os.path.join(output_folder, "model.zip")
    download_file(model_url, model_zip_path)
    extract_zip(model_zip_path, output_folder)
    os.remove(model_zip_path)  # Șterge fișierul ZIP

if __name__ == "__main__":
    main()
