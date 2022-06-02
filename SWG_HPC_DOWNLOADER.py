# %%
import json
import random
import numpy as np
import shutil
import os
import glob
from pathlib import Path
from operator import itemgetter 
from itertools import groupby
from os.path import exists
from urllib.parse import urlparse


random.seed(42)

# %%
metadata_path = '/home/mcs001/20204222/datasets/swg_camera_traps.bounding_boxes.with_species/swg_camera_traps.bounding_boxes.with_species.json'
# base_url = "https://lilablobssc.blob.core.windows.net/swg-camera-traps/"
# downloader = {'sas_url': 'https://lilablobssc.blob.core.windows.net/swg-camera-traps',
#                 'filenames' : [] }
lila_local_base = r'/home/mcs001/20204222/datasets/swg/images'

# %%
NUM_EMPTY, NUM_NONEMPTY = 10, 10

# %%
with open(metadata_path) as f:
    d = json.load(f)

# %%
for i in range(len(d['images'])):
    d['images'][i]['image_id'] = d['images'][i].pop('id')

my_id = itemgetter('image_id')
meta_anno = []

for k, v in groupby(sorted((d['annotations'] + d['images']), key=my_id), key=my_id):
    meta_anno.append({key:val for d in v for key, val in d.items()})

# %%
private_ids = set()
for idx, image in enumerate(meta_anno):
    if 'private' in image['file_name']:
        private_ids.add(image['image_id'])
print('number of private ids: {}'.format(len(private_ids)))
meta_anno = [img for img in meta_anno if img.get('image_id') not in private_ids]


corrupt_ids = set()
for idx, image in enumerate(meta_anno):
    if image['corrupt']:
        corrupt_ids.add(image['image_id'])
meta_anno = [img for img in meta_anno if img.get('image_id') not in corrupt_ids]
print('number of corrupt ids: {}'.format(len(corrupt_ids)))


category_remove_ids = set()
for idx, image in enumerate(meta_anno):
    try:
        if image['category_id'] in (1,2): # tags 'ignore' and 'blurred' removed.
            category_remove_ids.add(image['image_id'])
    except KeyError: 
            category_remove_ids.add(image['image_id'])
meta_anno = [img for img in meta_anno if img.get('image_id') not in category_remove_ids]
print('number of images removed based on cats and keyerror: {}'.format(len(category_remove_ids)))

# %%
def filter_species():
    pass

def filter_locations():
    pass

# %%
def gen_dataset(d, remove_missing_id, n_empty= 1000, n_nempty=1000):
    n_empty_images_per_dataset = n_empty
    n_non_empty_images_per_dataset = n_nempty

    category_id_to_name = {c['id']:c['name'] for c in d['categories']}
    category_name_to_id = {c['name']:c['id'] for c in d['categories']}


    human_category_id = category_name_to_id['human'] if 'human' in category_name_to_id.keys() else -1 # filter out humans


    if 'empty' not in category_name_to_id:
        print('Warning: no empty images available for {}'.format('dataset'))
        empty_category_id = -1
        empty_annotations = []
        empty_annotations_to_download = []
    else:
        empty_category_id = category_name_to_id['empty']        
        empty_annotations = [ann for ann in d['annotations'] if ann['category_id'] == empty_category_id and ann['image_id'] not in remove_missing_id]
        empty_annotations_to_download = random.sample(empty_annotations, n_empty_images_per_dataset)        
        
    non_empty_annotations = [ann for ann in d['annotations'] if ann['category_id'] not in (empty_category_id, human_category_id) and ann['image_id'] not in remove_missing_id]

    non_empty_annotations_to_download = random.sample(non_empty_annotations, n_non_empty_images_per_dataset)
    annotations_to_download = empty_annotations_to_download + non_empty_annotations_to_download
    image_ids_to_download = set([ann['image_id'] for ann in annotations_to_download])
    assert len(image_ids_to_download) == len(set(image_ids_to_download))

    images_to_download = []
    for im in d['images']:
        if im['image_id'] in image_ids_to_download:
            images_to_download.append(im)
    assert len(images_to_download) == len(image_ids_to_download)
    
    return images_to_download

# %%
images_to_download = gen_dataset(d, [-1], NUM_EMPTY, NUM_NONEMPTY)
train, validate, test = np.split(images_to_download, [int(.8*len(images_to_download)), int(.9*len(images_to_download))])

# %%

# %%
setpaths = {}
for dataset, setname in zip((train, validate, test), ("train", "val", "test")):
    sas_url =  'https://lilablobssc.blob.core.windows.net/swg-camera-traps'
    filenames = []
    for im in dataset:
        filenames = [im['file_name'] for im in dataset] # if im['id'] in image_ids_of_interest]


    if '?' in sas_url:
        base_url = sas_url.split('?')[0]        
        sas_token = sas_url.split('?')[1]
        assert not sas_token.startswith('?')
    else:
        sas_token = ''
        base_url = sas_url
        
    assert not base_url.endswith('/')

    p = urlparse(base_url)
    account_path = p.scheme + '://' + p.netloc
    assert account_path == 'https://lilablobssc.blob.core.windows.net'

    container_and_folder = p.path[1:]
    
    if len(container_and_folder.split('/')) == 2:
        container_name = container_and_folder.split('/')[0]
        folder = container_and_folder.split('/',1)[1]
        filenames = [folder + '/' + s for s in filenames]
    else: 
        assert(len(container_and_folder.split('/')) == 1)
        container_name = container_and_folder

    container_sas_url = account_path + '/' + container_name
    if len(sas_token) > 0:
        container_sas_url += '?' + sas_token

    output_dir = os.path.join(lila_local_base, setname)
    setpaths[setname] = output_dir
    os.makedirs(output_dir,exist_ok=True)

    # The container name will be included because it's part of the file name
    container_output_dir = output_dir # os.path.join(output_dir,container_name)

    os.makedirs(container_output_dir,exist_ok=True)

    # Write out a list of files, and use the azcopy "list-of-files" option to download those files
    # this azcopy feature is unofficially documented at https://github.com/Azure/azure-storage-azcopy/wiki/Listing-specific-files-to-transfer
    az_filename = os.path.join(output_dir, 'filenames_{}.txt'.format('TRAIN'.lower().replace(' ','_')))
    with open(az_filename, 'w') as f:
        for fn in filenames:
            f.write(fn.replace('\\','/') + '\n')
            
    cmd = '/home/mcs001/20204222/azcopy_linux_amd64_10.15.0/azcopy cp "{0}" "{1}" --list-of-files "{2}"'.format(
            container_sas_url, container_output_dir, az_filename)            

    # import clipboard; clipboard.copy(cmd)

    os.system(cmd)


# %%
for p in setpaths.values():
    for f in glob.glob(p + '/**/**/**/**/**/*.jpg', recursive=True):
        fnew = f[f.find('public'):].replace('/','-')
        try:
            shutil.move(f, p + '/' + fnew)
        except FileNotFoundError:
            continue


# %%
def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier


def createLabelsSingle(imageList, basedir, labeldirname, metadata_full):
    # For single objects only

    basedir = basedir.parent
    os.makedirs(str(basedir) + "/labels/" + labeldirname,exist_ok=True)

    ids = [i.get('image_id') for i in imageList]
    # generate lookup for bbox and category id based on image id


    print("!WARNING: hardcoded fix for islands dataset")

    lookup = {}
    for meta in metadata_full["annotations"]:
        if meta["image_id"] not in ids: continue

        bb = [0, 0, 0, 1] #TODO this is hardcoded fix/default for the islands dataset 

        try:
            bb = meta['bbox']
        except KeyError:
            if meta['category_id'] != 0:
                raise KeyError('Keyerror on boundingbox but not an empty image!')

        lookup[meta['image_id']] = {"bbox": bb, "category_id": meta["category_id"]}


    for im in imageList:

        ann = lookup.get(im['image_id'])

        dw = 1. / im['width']
        dh = 1. / im['height']
        
        
        filename = im['file_name'].replace(".jpg", ".txt").replace("/", "-")
        # print(Path(basedir).parent.__str__() + "/labels/" + labeldirname + filename, "a")
        with open(str(basedir) + "/labels/" + labeldirname + filename, "a") as myfile:
            xmin = ann["bbox"][0]
            ymin = ann["bbox"][1]
            xmax = ann["bbox"][2] + ann["bbox"][0]
            ymax = ann["bbox"][3] + ann["bbox"][1]
            
            x = (xmin + xmax)/2
            y = (ymin + ymax)/2
            
            w = xmax - xmin
            h = ymax-ymin
            
            x = x * dw
            w = w * dw
            y = y * dh
            h = h * dh
            
            mystring = str(str(ann['category_id']) + " " + str(truncate(x, 7)) + " " + str(truncate(y, 7)) + " " + str(truncate(w, 7)) + " " + str(truncate(h, 7)))
            myfile.write(mystring)
            myfile.write("\n")

        myfile.close()

# %%
for dataset, setname in zip((train, validate, test), ("train", "val", "test")):
    setpath = setpaths.get(setname) 
    createLabelsSingle(dataset, Path(setpath).parent, setname + '/', d)

# %%



