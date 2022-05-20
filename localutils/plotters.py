from importlib.resources import path
from matplotlib import pyplot as plt
import torchvision

def plotTensor(imtensor):
  if len(imtensor.shape) == 4: #for batches
    fig, ax = plt.subplots(figsize=(16, 16), dpi=80)
    grid_img = torchvision.utils.make_grid(imtensor, nrow=4)
    ax.imshow(grid_img.permute(1,2,0))
  elif len(imtensor.shape) == 3:
    fig, ax = plt.subplots(figsize=(10, 10), dpi=80)
    ax.imshow(imtensor.permute(1,2,0))
  plt.show()
  return


from PIL import Image
import matplotlib

def boundingBox(img: str, bbox: list[3]):
  im = Image.open(img)

  # Create figure and axes
  fig, ax = plt.subplots()

  # Display the image
  ax.imshow(im)

  # Create a Rectangle patch
  rect = matplotlib.patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', facecolor='none')
  plt.text(bbox[0], bbox[1],'lalala')

  # Add the patch to the Axes
  ax.add_patch(rect)

  plt.show()


def boundingBox2(img: str, bbox: list[3], label: str ='label'):
  im = Image.open(img)

  # Create figure and axes
  fig, ax = plt.subplots()

  # Display the image
  ax.imshow(im)

  # Create a Rectangle patch
  rect = matplotlib.patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', facecolor='none')
  # Add the patch to the Axes
  ax.add_patch(rect)
  rect2 = matplotlib.patches.Rectangle((bbox[0], bbox[1]-100), 500, 100, facecolor='r')
  ax.add_patch(rect2)

  plt.text(bbox[0]+20, bbox[1]-40, label, fontdict={'weight': 'bold', 'size': 10})


  plt.show()

