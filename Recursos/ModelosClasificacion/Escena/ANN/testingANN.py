import argparse
import sys
import cv2
import numpy as np
import time

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

IMAGE_WIDTH  = 64
IMAGE_HEIGHT = 128
IMAGE_DEPTH  = 1
IMAGE_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT

parser = argparse.ArgumentParser()
parser.add_argument(
    '--image',
    default="/Users/user/Desktop/Recursos/BancosImagenes/Daimler/Prueba/No_Persona/neg1.jpg",
    type=str,
    help='Absolute path to image file.')
parser.add_argument(
    '--num_top_predictions',
    type=int,
    default=2,
    help='Display this many predictions.')
parser.add_argument(
    '--graph',
    default='C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/ModelosClasificacion/ModelosEntrenados/ANN/',
    type=str,
    help='Absolute path to graph file (.pb)')
parser.add_argument(
    '--banco',
    default='Daimler',
    type=str,
    help='Absolute path to graph file (.pb)')
parser.add_argument(
    '--labels',
    default='C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/ModelosClasificacion/ModelosEntrenados/CNN/Daimler/output_labels.txt',
    type=str,
    help='Absolute path to labels file (.txt)')
parser.add_argument(
    '--output_layer',
    type=str,
    default='Probabilidad:0',
    help='Name of the result operation')
parser.add_argument(
    '--input_layer',
    type=str,
    default='Imagenes:0',
    help='Name of the input operation')

def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]

def run_graph(folder, labels, input_layer_name, output_layer_name, num_top_predictions,filename):
  with tf.Session() as sess:
    saver = tf.train.import_meta_graph(filename)
    saver.restore(sess,tf.train.latest_checkpoint(FLAGS.graph+FLAGS.banco+"/"))

    images = []

    image = cv2.imread(folder,0)
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, (1.0/255.0) - 0.5)
    x_batch = images.reshape(1, IMAGE_PIXELS)

    graph = tf.get_default_graph()

    output_tensor = graph.get_tensor_by_name(output_layer_name)

    input_tensor = graph.get_tensor_by_name(input_layer_name)
    feed_dict_testing = {input_tensor: x_batch}
    predictions, = sess.run(output_tensor, feed_dict=feed_dict_testing)

    top_k = predictions.argsort()[-num_top_predictions:][::-1]

    file1 = open("C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/ModelosClasificacion/Escena/ANN/Predicciones.txt","a")
    file2 = open("C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/ModelosClasificacion/Escena/ANN/Scores.txt","a")
    labels = labels[::-1]
    for node_id in top_k:
      human_string = labels[node_id]
      score = predictions[node_id]
      print('%s (score = %.5f)' % (human_string, score))
      if score>0.500 :
          print('Gano la categoria',human_string)
          if human_string=="persona":
              file1.write("1"+"\n")
              file2.write(str(score)+"\n")
          else:
              file1.write("0"+"\n")
              file2.write(str(score)+"\n")
    file1.close()
    file2.close()
    return 0


def main(argv):

  """Runs inference on an image."""
  if argv[1:]:
    raise ValueError('Unused Command Line Args: %s' % argv[1:])

  if not tf.gfile.Exists(FLAGS.image):
    tf.logging.fatal('image file does not exist %s', FLAGS.image)

  if not tf.gfile.Exists(FLAGS.labels):
    tf.logging.fatal('labels file does not exist %s', FLAGS.labels)

  if not tf.gfile.Exists(FLAGS.graph+FLAGS.banco+"/ANN.meta"):
    tf.logging.fatal('graph file does not exist %s', FLAGS.graph+FLAGS.banco+"/ANN.meta")

  # load labels
  labels = load_labels(FLAGS.labels)

  run_graph(FLAGS.image, labels, FLAGS.input_layer, FLAGS.output_layer, FLAGS.num_top_predictions,FLAGS.graph+FLAGS.banco+"/ANN.meta")


if __name__ == '__main__':
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=sys.argv[:1]+unparsed)
