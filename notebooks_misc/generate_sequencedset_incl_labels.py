#%%
import json
from pathlib import Path
import glob 
import os
import tqdm
import pickle
import concurrent_hists

#%%
picklepath = "C:/temp/islandsdataset/pickle/train5.pkl"
impath = "C:/temp/islandsdataset/images/islands64val"
labelpath = "C:/temp/islandsdataset/labels/islands64val"

#%%
imnames = [os.path.split(i)[1].split('.jpg')[0] for i in glob.glob(impath + '/*.jpg', recursive=True)]
# %%
metadata_path = '../data/islands/metadata.json'
with open(metadata_path) as f:
    d = json.load(f)

#%%
for anno in d['annotations']:
    if anno.get('category_id') == 6:
        anno['category_id'] = 1
labellookup = {i.get('image_id'): i.get('category_id') for i in d['annotations']}
# %%
for imn in imnames: 
    cat = labellookup.get(imn)
    with open(labelpath + '/' + imn + '.txt', 'a') as f:
        mystring = str(str(cat) + " 0.5 0.5 1 1") 
        f.write(mystring)
        f.write("\n")
    f.close()

# %%
