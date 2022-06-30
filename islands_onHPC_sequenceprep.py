#%%
#!pip install pickle5
import pickle5 
import shutil

#%%
picklepath = "/home/mcs001/20204222/datasets/sequence/pickles/train10.pk"
impath = "/home/mcs001/20204222/datasets/sequence/images_full/images/train10/" 
labelpath = "/home/mcs001/20204222/datasets/sequence/images_full/labels/train10/"
basepath = "/home/mcs001/20204222/datasets/islandsraw/images/"

# %%
with open(picklepath, 'rb') as f:
    meta_anno = pickle5.load(f)

#%%
def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

#%%
for im in meta_anno[:10]:    

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