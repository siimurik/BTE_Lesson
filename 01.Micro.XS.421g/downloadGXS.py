import requests
import os
import zipfile

# Dictionary of file names and their corresponding URLs
file_urls = {
    "B_010.GXS": "https://www-nds.iaea.org/ads/GENDF/B_010G.ZIP",
    "B_011.GXS": "https://www-nds.iaea.org/ads/GENDF/B_011G.ZIP",
    "H_001.GXS": "https://www-nds.iaea.org/ads/GENDF/H_001G.ZIP",
    "O_016.GXS": "https://www-nds.iaea.org/ads/GENDF/O_016G.ZIP",
    "U_235.GXS": "https://www-nds.iaea.org/ads/GENDF/U_235G.ZIP",
    "U_238.GXS": "https://www-nds.iaea.org/ads/GENDF/U_238G.ZIP",
    "ZR090.GXS": "https://www-nds.iaea.org/ads/GENDF/ZR090G.ZIP",
    "ZR091.GXS": "https://www-nds.iaea.org/ads/GENDF/ZR091G.ZIP",
    "ZR092.GXS": "https://www-nds.iaea.org/ads/GENDF/ZR092G.ZIP",
    "ZR094.GXS": "https://www-nds.iaea.org/ads/GENDF/ZR094G.ZIP",
    "ZR096.GXS": "https://www-nds.iaea.org/ads/GENDF/ZR096G.ZIP"
}

# Create a directory to save the downloaded files
directory = "GXS_files"
if not os.path.exists(directory):
    os.makedirs(directory)

# Download each ZIP file and extract the GXS file
for file_name, file_url in file_urls.items():
    # Send a GET request to download the ZIP file
    response = requests.get(file_url, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        # Save the ZIP file
        zip_file_path = os.path.join(directory, file_name + ".zip")
        with open(zip_file_path, "wb") as zip_file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    zip_file.write(chunk)
        
        # Extract the GXS file from the ZIP file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(directory)
        
        # Remove the ZIP file
        os.remove(zip_file_path)
        
        print(f"Downloaded and extracted {file_name}")
    else:
        print(f"Failed to download {file_name}")
