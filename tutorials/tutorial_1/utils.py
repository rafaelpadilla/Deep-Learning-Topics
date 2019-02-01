import os
import urllib.request

def Download_file(url, dir_destination=None, file_name=None, quiet=False):
    
    if file_name == None:
        file_name = url[url.rfind('/')+1:]
    # If directory does not exist, create it
    if not os.path.exists(dir_destination):
        os.makedirs(dir_destination)
    # Set path of the file
    full_path_file = os.path.join(dir_destination,file_name)
    if quiet == False:
        print('Trying to reach %s' % url)
    response = urllib.request.urlopen(url)
    # Read as bytes data into memory
    data = response.read()
    with open(full_path_file, 'wb') as out_file:
        a = out_file.write(data)

current_dir = os.path.dirname(os.path.realpath(__file__))

url = 'https://download.pytorch.org/tutorial/hymenoptera_data.zip'
dir_to_save = os.path.join(current_dir,'../datasets')
Download_file(url, dir_to_save, 'hymenoptera_data.zip')


