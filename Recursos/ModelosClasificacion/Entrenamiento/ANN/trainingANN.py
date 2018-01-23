import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

import numpy
import numpy as np
import math
import argparse
import sys
from PIL import Image
from six.moves import xrange
import time

IMAGE_WIDTH  = 64
IMAGE_HEIGHT = 128
IMAGE_DEPTH  = 1
IMAGE_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT
NUM_CLASSES  = 2

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dir',
    default="/Users/user/Desktop/Recursos/BancosImagenes/",
    type=str,
    help='Directorio a los bancos de imagenes')
parser.add_argument(
    '--banco',
    default="Daimler",
    type=str,
    help='Display this many predictions.')
parser.add_argument(
    '--learning',
    type=float,
    default=0.02,
    help='Display this many predictions.')
parser.add_argument(
    '--epocas',
    type=int,
    default=100,
    help='Display this many predictions.')
parser.add_argument(
    '--numExamPos',
    type=int,
    default=5000,
    help='Display this many predictions.')
parser.add_argument(
    '--numExamNeg',
    type=int,
    default=5000,
    help='Display this many predictions.')
parser.add_argument(
    '--batch',
    type=int,
    default=20,
    help='Display this many predictions.')
parser.add_argument(
    '--numNeurons',
    type=int,
    default=20,
    help='Display this many predictions.')
parser.add_argument(
    '--dirModelo',
    default="",
    type=str,
    help='Display this many predictions.')

def read_my_list( numNeg, numPos, folder ):

    filenames = []
    labels    = []

    if folder=="Prueba":
        numNeg=int(numNeg*0.2)
        numPos=int(numPos*0.2)

    for num in range( 1, numPos+1 ):

        filenames.append( FLAGS.dir + FLAGS.banco + "/" + folder + "/Persona/" + name_si( num ) + ".jpg" )
        #labels[ ( num - minId ) * 2 ][ 1 ] = 1
        labels.append( int( 1 ) )

    for num in range( 1, numNeg+1 ):
        filenames.append( FLAGS.dir + FLAGS.banco + "/" + folder + "/No_Persona/" + name_no( num ) + ".jpg" )
        #labels[ ( ( num - minId ) * 2 ) + 1 ][ 0 ] = 1
        labels.append( int( 0 ) )

    print( "number of labels: " + str( len( labels ) ) )
    print( "number of images: " + str( len( filenames ) ) )
    return filenames, labels

def num_name( id ):

    ret = str( id )
    while ( len( ret ) < 5 ):
        ret = "0" + ret;

    return ret;

def name_si( id ):

    ret = str( id )
    ret = "Pos" + ret;

    return ret;

def name_no( id ):

    ret = str( id )
    ret = "neg" + ret;

    return ret;

def read_images_from_disk(input_queue):

    label = input_queue[1]
    print( "-------------------------------------------------------------------" )
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_jpeg( file_contents, channels = 1 )
    example = tf.image.resize_images(images = example,size = [IMAGE_HEIGHT,IMAGE_WIDTH])
    example = tf.reshape( example, [ IMAGE_PIXELS ] )
    example.set_shape( [ IMAGE_PIXELS ] )
    example = tf.cast( example, tf.float32 )
    example = tf.cast( example, tf.float32 ) * ( 1. / 255 ) - 0.5

    label = tf.cast( label, tf.int64 )

    label = tf.one_hot( label, 2, 1, 0 )
    label = tf.cast( label, tf.float32 )
    print( "Fin lectura imagenes en disco" )
    print( "-------------------------------------------------------------------" )
    return  example, label

def fill_feed_dict(image_batch, label_batch, imgs, lbls):
  feed_dict = {
      imgs: image_batch,
      lbls: label_batch,
  }
  return feed_dict

def main(argv):
    tiempo_inicial = time.time()

    # config
    learning_rate = FLAGS.learning
    training_epochs = FLAGS.epocas
    num_pos = FLAGS.numExamPos
    num_neg = FLAGS.numExamNeg
    BATCH_SIZE    = FLAGS.batch

    # Network Parameters
    n_hidden_1 = FLAGS.numNeurons
    n_hidden_2 = n_hidden_1
    # input images
    # None -> batch size can be any size, 784 -> flattened mnist image
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_PIXELS], name="Imagenes")
    # target 10 output classes
    y_ = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES], name="y-input")

    #Model definition
    wh1 = tf.Variable(tf.random_normal([IMAGE_PIXELS, n_hidden_1]), name = "wh1")
    wh2 = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name = "wh2")
    wout = tf.Variable(tf.random_normal([n_hidden_2, NUM_CLASSES]), name = "wout")

    bh1 = tf.Variable(tf.random_normal([n_hidden_1]), name = "bh1")
    bh2 = tf.Variable(tf.random_normal([n_hidden_2]), name = "bh2")
    bout =  tf.Variable(tf.random_normal([NUM_CLASSES]), name = "bout")

    # Hidden layer with a given function activation
    layer_1 = tf.add(tf.matmul(x, wh1), bh1)
    layer_1 = tf.nn.sigmoid(layer_1)

    # Hidden layer with a given function activation
    layer_2 = tf.add(tf.matmul(layer_1, wh2), bh2)
    layer_2 = tf.nn.sigmoid(layer_2)

    # Output layer with linear activation
    y = tf.add(tf.matmul(layer_2, wout),bout, name = "Salida")

    y_pred = tf.nn.softmax(y,name="Probabilidad")

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Accuracy
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('Accuracy',accuracy)

    # DATA FOR TRAINING
    # get filelist and labels for training
    print( "LECTURA IMAGENES PARA ENTRENAMIENTO")
    image_list, label_list = read_my_list( int(num_neg), int(num_pos), "Entrenamiento" )

    # create queue for training
    input_queue = tf.train.slice_input_producer( [ image_list, label_list ])

    # read files for training
    image, label = read_images_from_disk( input_queue )

    # `image_batch` and `label_batch` represent the "next" batch
    # read from the input queue.
    image_batch, label_batch = tf.train.batch( [ image, label ], batch_size = BATCH_SIZE )

    # DATA FOR TESTING
    # get filelist and labels for tESTING
    print( "LECTURA IMAGENES PARA TEST")
    image_list_test, label_list_test = read_my_list(int(num_neg), int(num_pos), "Prueba" )

    # create queue for training
    input_queue_test = tf.train.slice_input_producer( [ image_list_test, label_list_test ])

    # read files for training
    image_test, label_test = read_images_from_disk( input_queue_test )

    # read from the input queue.
    image_batch_test, label_batch_test = tf.train.batch( [ image_test, label_test ], batch_size = BATCH_SIZE )

    with tf.Session() as sess:

        # variables need to be initialized before we can use them
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        merged_summary = tf.summary.merge_all()
        train_writer=tf.summary.FileWriter(FLAGS.dirModelo + 'ANN/' + FLAGS.banco+"/Logs/Entrenamiento")
        validation_writer=tf.summary.FileWriter(FLAGS.dirModelo + 'ANN/' + FLAGS.banco+"/Logs/Validacion")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # perform training cycles
        for epoch in range(training_epochs):

            batch_count = int((num_pos+num_neg)/BATCH_SIZE)

            for i in range(1):
                imgs, lbls = sess.run([image_batch, label_batch])
                sess.run([optimizer], feed_dict={x:imgs, y_:lbls})
            if epoch % 1 ==0:
                print("Epoca: ", epoch)
                print ("Exactitud entrenamiento: ", accuracy.eval(feed_dict={x: imgs , y_: lbls}))
                imgs_test, lbls_test = sess.run([image_batch_test, label_batch_test])
                print ("Exactitud validacion: ", accuracy.eval(feed_dict={x: imgs_test , y_: lbls_test}))
                t = sess.run(merged_summary, feed_dict={x: imgs , y_: lbls})
                train_writer.add_summary(t,epoch)
                v = sess.run(merged_summary, feed_dict={x: imgs_test , y_: lbls_test})
                validation_writer.add_summary(v,epoch)
            else:
                print("Epoca: ", epoch)
                print ("Exactitud entrenamiento: ", accuracy.eval(feed_dict={x: imgs , y_: lbls}))
                t = sess.run(merged_summary, feed_dict={x: imgs , y_: lbls})
                train_writer.add_summary(t,epoch)

            '''
            # number of batches in one epoch
            batch_count = int((num_pos+num_neg)/BATCH_SIZE)

            for i in range(batch_count):

                imgs, lbls = sess.run([image_batch, label_batch])

                sess.run([optimizer], feed_dict={x:imgs, y_:lbls})

            print("Epoca: ", epoch)
            imgs_test, lbls_test = sess.run([image_batch_test, label_batch_test])
            print ("Exactitud: ", accuracy.eval(feed_dict={x: imgs_test , y_: lbls_test}))
            #print ("Salida: ", correct_prediction.eval(feed_dict={x: imgs_test , y_: lbls_test}))
            '''


        if not os.path.exists(FLAGS.dirModelo + 'ANN/' + FLAGS.banco):
            os.makedirs(FLAGS.dirModelo + 'ANN/' + FLAGS.banco)

        saver.save(sess, FLAGS.dirModelo + '/ANN/' + FLAGS.banco + "/ANN")
        coord.request_stop()
        coord.join(threads)

        #input('Press enter to continue: ')

        tiempo_final = time.time()

        tiempo_ejecucion = tiempo_final - tiempo_inicial

        fichero = open("Tiempo.txt","w")

        fichero.write(str(tiempo_ejecucion))

        print ('El tiempo de ejecucion fue:',tiempo_ejecucion)


if __name__ == '__main__':
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=sys.argv[:1]+unparsed)
