=============
v4lbodykeying
=============

A fun project to separate persons from background in a Video4Linux stream. The
result is provided as virtualVideo4Linux device and hence can be used to bring
"virtual background" support to any video-streaming app. And all of that without
a green screen.

The idea (conceptionally and implementation-wise) is taken from Benjamin Elder's
amazing blog post `"Open Source Virtual Background"
<https://elder.dev/posts/open-source-virtual-background/>`_. There are a few key
differences though:

* No NodeJS helper application - (almost) all in Python
* `FFmpeg`_ is used for collecting and playing back frames

What stays the same?

* `v4l2loopback`_ to provide the modified stream as video device
* `BodyPix`_ model to provide the keying feature itself
* `OpenCV`_ for some image manipulations


Requirements
============

* Python 3.7
* `FFmpeg`_ with v4l support
* `Tensorflow <https://www.tensorflow.org/api_docs/python/tf>`_
* `OpenCV 4 <https://opencv.org/>`_
* A BodyPix model in Tensorflow's "frozen graph" format (ProtoBuf)


Obtaining the model
===================

The model can be downloaded in TensorflowJS format from Google's servers::

  curl -O https://storage.googleapis.com/tfjs-models/savedmodel/bodypix/mobilenet/float/075/model-stride16.json
  curl -O https://storage.googleapis.com/tfjs-models/savedmodel/bodypix/mobilenet/float/075/group1-shard1of2.bin
  curl -O https://storage.googleapis.com/tfjs-models/savedmodel/bodypix/mobilenet/float/075/group1-shard2of2.bin

One can then use `tfjs-to-tf <https://github.com/patlevin/tfjs-to-tf>`_ to
convert it into a Tensorflow-understandable model.


.. _BodyPix: https://github.com/tensorflow/tfjs-models/tree/master/body-pix
.. _FFmpeg: https://www.ffmpeg.org/
.. _OpenCV: https://opencv.org/
.. _v4l2loopback: https://github.com/umlaeute/v4l2loopback
