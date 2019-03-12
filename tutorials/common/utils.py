import os
import tarfile
import zipfile
from six.moves import urllib


def fetch_dataset(url, destination_folder, decompress=False, del_file_afterwards=False, quiet=False):
    if not os.path.isdir(destination_folder):
        os.makedirs(destination_folder)
    file_name = url[url.rfind('/')+1:]
    destination_file = os.path.join(destination_folder, file_name)
    if not quiet:
        print(f'Downloading file from {url}')
    try:
        urllib.request.urlretrieve(url, destination_file)
        if not quiet:
            print(f'File successfully downloaded from {url}')
    except:
        print(f'Error downloading file from {url}')
        return False, ''
    if not decompress:
        return True, ''
    ending_folder = ''
    if file_name.endswith('.tar') or file_name.endswith('.tar.gz') or file_name.endswith('.tgz'):
        ending_folder = destination_file.replace('.tar','/').replace('.tar.gz','/'),replace('.tgz','/')
        if not quiet:
            print(f'Decompressing {file_name} with tarfile into {ending_folder}.')
        tgz_file = tarfile.open(destination_file)
        tgz_file.extractall(path=destination_folder)
        tgz_file.close()
    elif file_name.endswith('.zip'):
        ending_folder = destination_file.replace('.zip','/')
        if not quiet:
            print(f'Decompressing {file_name} with zipfile into {ending_folder}')
        zip_file = zipfile.ZipFile(destination_file)
        zip_file.extractall(destination_folder)
        zip_file.close()
    else:
        if not quiet:
            print('Correct decompression tool not found.') 
        if del_file_afterwards:
            a = os.remove(destination_file) 
        if not quiet and a:
            print(f'File {file_name} deleted sucessfully.') 
        return False, ending_folder
    
    if del_file_afterwards:
        try:
            os.remove(destination_file)
            if not quiet:
                print(f'File {file_name} deleted sucessfully.') 
        except:
            print(f'Error deleting {file_name}.') 
    return True, ending_folder



