#%%
#!pip install pickle5
import pickle
import shutil

#%%
basepath = "C:\Projects\wild\data\islands\images\images\\"
picklepath = "C:/temp/islandsdataset/pickles/val20.pk"
impath = "C:/temp/islandsdataset/images/val20/"
labelpath = "C:/temp/islandsdataset/labels/val20/"
# %%
with open(picklepath, 'rb') as f:
    meta_anno = pickle.load(f)

#%%
def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

#%%
for im in meta_anno:    

        imid = im.get('image_id')
        ann = im.get('category_id')

        # quick fix for faulty label 
        if ann == 6: 
            ann = 1

        dw = 1. / im['width']
        dh = 1. / im['height']

        shutil.copy(basepath + im['file_name'], impath + im['image_id'] + '.jpg')
        
        
        filename = im['image_id'] + '.txt'
        # print(Path(basedir).parent.__str__() + "/labels/" + labeldirname + filename, "a")
        with open(labelpath + filename, "a") as myfile:
            try:
                xmin = im["bbox"][0]
                ymin = im["bbox"][1]
                xmax = im["bbox"][2] + im["bbox"][0]
                ymax = im["bbox"][3] + im["bbox"][1]
                
                x = (xmin + xmax)/2
                y = (ymin + ymax)/2
                
                w = xmax - xmin
                h = ymax-ymin
                
                x = x * dw
                w = w * dw
                y = y * dh
                h = h * dh
            except KeyError:
                x, y = 0, 0
                w, h = 0.5, 0.5
            
            mystring = str(str(ann) + " " + str(truncate(x, 7)) + " " + str(truncate(y, 7)) + " " + str(truncate(w, 7)) + " " + str(truncate(h, 7)))
            myfile.write(mystring)
            myfile.write("\n")

        myfile.close()
# %%
