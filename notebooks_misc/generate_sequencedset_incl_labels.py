#%%
import json
from pathlib import Path
import glob 
import os
from tqdm import tqdm
import pickle
# from backgroundsubseq import generate_boxed_by_sequence
import backgroundsubseq
import cv2

#%%
import importlib
importlib.reload(backgroundsubseq)

#%%
datasetpath = "C:\Projects\wild\data\islands\images\images\\"
picklepath = "C:/temp/islandsdataset/pickles/train10.pk"
impath = "C:/temp/islandsdataset/images/64xtrain10/"
labelpath = "C:/temp/islandsdataset/labels/"

#%%
with open(picklepath, 'rb') as f:
    meta_anno = pickle.load(f)

#%%
sequences = list(set([i.get('seq_id') for i in meta_anno]))
labellookup = {i.get('image_id'): str(i.get('category_id')).replace('6','1') for i in meta_anno}
imgs_seq_lookup = {}
for ma in meta_anno:
    imgs_seq_lookup.setdefault(ma.get('seq_id','empty'),[]).append(ma)
for sequence in tqdm(sequences[811:814]):
    filenames = [i.get('file_name') for i in imgs_seq_lookup.get(sequence)]
    ids = [i.get('image_id') for i in imgs_seq_lookup.get(sequence)]
    p = lambda x: str(Path(datasetpath) / x)
    paths = [p(x) for x in filenames]
    imgs = backgroundsubseq.generate_boxed_by_sequence(paths, 64)
    sequences.append(imgs)
    for im, fn in zip(imgs, ids):
        cv2.imwrite(impath + fn + '.jpg', im)
        with open(labelpath + fn + '.txt', 'a') as f:
            mystring = str(str(labellookup.get(fn)) + " 0.5 0.5 1 1") 
            f.write(mystring)
            f.write("\n")

# #%%
# for anno in d['annotations']:
#     if anno.get('category_id') == 6:
#         anno['category_id'] = 1
# # %%
# for imn in imnames: 
#     cat = labellookup.get(imn)
#     with open(labelpath + '/' + imn + '.txt', 'a') as f:
#         mystring = str(str(cat) + " 0.5 0.5 1 1") 
#         f.write(mystring)
#         f.write("\n")
#     f.close()

# # %%

# %%
