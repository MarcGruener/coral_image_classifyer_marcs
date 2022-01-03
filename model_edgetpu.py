import argparse
import os
import pathlib
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
from PIL import Image


def main():
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
    '-i', '--input', required=True, help='Image to be classified.')
  parser.add_argument(
    '-m', '--model', help='File path of .tflite file.')
  parser.add_argument(
    '-l', '--labels', help='File path of labels file.')
    
  args = parser.parse_args()


  # Specify the TensorFlow model, labels, and image
  script_dir = pathlib.Path(__file__).parent.absolute()
  model_file = os.path.join(script_dir, 'model_edgetpu.tflite')
  label_file = os.path.join(script_dir, 'labels.txt')
  image_file = os.path.join(script_dir, args.input)

  # Initialize the TF interpreter
  interpreter = edgetpu.make_interpreter(model_file)
  interpreter.allocate_tensors()

  # Resize the image
  size = common.input_size(interpreter)
  image = Image.open(image_file).convert('RGB').resize(size, Image.ANTIALIAS)
  image.resize((224,224))
  image.transpose(Image.FLIP_LEFT_RIGHT)

  # Run an inference
  common.set_input(interpreter, image)
  interpreter.invoke()
  classes = classify.get_classes(interpreter, top_k=4)

  # Print the result
  labels = dataset.read_label_file(label_file)
  for c in classes:
    print('%s: %.5f' % (labels.get(c.id, c.id), c.score))

if __name__ == '__main__':
    main()
