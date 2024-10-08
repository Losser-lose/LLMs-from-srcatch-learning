import urllib.request
import zipfile
import os
from pathlib import Path

def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    # Downloading the file
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    # Unzipping the file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    # Add .tsv file extension
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")
    
    
if __name__ == "__main__":
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "/Users/kingsun/Desktop/LLM/projects/my-projects/LLMs-from-srcatch-learning/data/sms_spam_collection/sms_spam_collection.zip"
    extracted_path = "/Users/kingsun/Desktop/LLM/projects/my-projects/LLMs-from-srcatch-learning/data/sms_spam_collection"
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"
    
    download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)