import os
import numpy as np
from PIL import Image
import tensorflow as tf

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 1280
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    # restore from pb file
    file_handle = open(tarball_path, 'rb')
    graph_def = tf.GraphDef.FromString(file_handle.read())

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map


def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap


def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]


LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

LABEL_NAMES_CITYSCAPES = np.asarray([
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation','terrain','sky','person','rider',
    'car','truck','bus','train','motorcycle','bicycle',
])

LABEL_NAMES_NAVI = np.asarray([
    'road', 'sidewalk', 'pole','traffic light', 'traffic sign', 'person', 'car','others',
])
                                     
FULL_LABEL_MAP = np.arange(len(LABEL_NAMES_NAVI)).reshape(len(LABEL_NAMES_NAVI), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

model_path = './models/model_8_1_0_multi.pb'
MODEL = DeepLabModel(model_path)
print('model loaded successfully!')

def predict(file_path):
  store_path = file_path.replace('images', 'results')
  try:
    img = Image.open(file_path)
    resized_img, seg_map = MODEL.run(img)
  except:
    print("error %s" % file_path)
  else:
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    im = Image.blend(resized_img, Image.fromarray(seg_image), 0.7)
    im.save(store_path)
