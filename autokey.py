import struct
import subprocess
import sys
import time
from threading import Condition, Lock, Thread

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio


MODEL_STRIDE = 16


def load_graph(frozen_graph_path):
    with tf.io.gfile.GFile(frozen_graph_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])

    def function_builder(inputs, outputs):
        return wrapped_import.prune(
            tf.nest.map_structure(wrapped_import.graph.as_graph_element, inputs),
            tf.nest.map_structure(wrapped_import.graph.as_graph_element, outputs))
    return (wrapped_import.graph, function_builder)


def _get_input_ops(graph):
    return (op for op in graph.get_operations() if op.op_def.name == "Placeholder")


def _get_output_ops(graph):
    ops = graph.get_operations()
    for (i, op) in enumerate(ops):
        if op.op_def.name not in {"Const", "Placeholder"}:
            if all(f"{op.name}:0" not in {x.name for x in other.inputs} for other in ops[i+1:]):
               yield op


class BodyPix:
    def __init__(self, predict_function):
        self._predict_function = predict_function

    @classmethod
    def from_path(cls, frozen_graph_path):
        (graph, factory) = load_graph(frozen_graph_path)
        input_ops = list(_get_input_ops(graph))
        if len(input_ops) != 1:
            raise ValueError(f'Expected 1 input op, got {len(input_ops)}')
        input_tensor_name = f"{input_ops[0].name}:0"
        return cls(factory(input_tensor_name, 'float_segments:0'))

    def segmentize(self, image, *, segmentation_threshold=0.5, internal_resolution=0.75):
        (img_height, img_width, _) = image.shape
        if img_height % MODEL_STRIDE != 0 or img_width % MODEL_STRIDE != 0:
            raise ValueError('Padding not supported')
        prepared_img = tf.image.resize(
            image,
            [int(img_height * internal_resolution / MODEL_STRIDE) * MODEL_STRIDE + 1,
             int(img_width * internal_resolution / MODEL_STRIDE) * MODEL_STRIDE + 1])
        (resized_height, resized_width, _) = prepared_img.shape
        segments = np.squeeze(self._predict(prepared_img), 0)
        resized_segments = tf.image.resize_with_pad(segments, img_height, img_width)
        return tf.dtypes.cast(
            tf.math.greater(tf.sigmoid(resized_segments), tf.constant(segmentation_threshold)),
            tf.float32)

    # XXX add typing hints (returns logits)
    def _predict(self, input):
        batch = tf.expand_dims(self._preprocess(input), 0)
        return self._predict_function(batch)

    def _preprocess(self, input):
        raise NotImplementedError


class MobileNetV1(BodyPix):
    def _preprocess(self, input):
        # Normalize the pixels [0, 255] to be between [-1, 1].
        return input / 127.5 - 1.0


class ResNet50(BodyPix):
    def _preprocess(self, input):
        # Add image net mean
        return input + tf.constant([-123.15, -115.90, -103.06])


class FrameGrabber:
    def __init__(self, device):
        self.device = device

    def __enter__(self):
        self._ffmpeg = subprocess.Popen(
            ["ffmpeg", '-f', 'v4l2', '-video_size', 'hd720', '-i', self.device,
             '-f', 'image2pipe', '-vcodec', 'bmp', '-r', '10', '-'],
            stdout=subprocess.PIPE)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._ffmpeg.kill()

    def run(self, ref):
        size_struct = struct.Struct('<I')
        while True:
            header = self._ffmpeg.stdout.read(6)
            if header[0:2] != b"BM":
                raise RuntimeError("Expected Bitmap header not found")
            (size, ) = size_struct.unpack_from(header, 2)
            bmp = tf.image.decode_bmp(header + self._ffmpeg.stdout.read(size - 6))
            ref.set(bmp)


class FrameWriter:
    def __init__(self, device):
        self.device = device

    def __enter__(self):
        self._ffmpeg = subprocess.Popen(
            ['ffmpeg', '-i', '-', '-pix_fmt', 'yuv420p', '-f', 'v4l2',
             '-video_size', 'hd720', self.device],
            stdin=subprocess.PIPE)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._ffmpeg.kill()

    def run(self, ref):
        while True:
            tensor = ref.get()
            as_uint8 = tf.dtypes.cast(tensor, tf.uint8)
            self._ffmpeg.stdin.write(tfio.image.encode_bmp(as_uint8).numpy())
            self._ffmpeg.stdin.flush()


class MRef:
    """
    A thread-safe mutable reference. It has two operations: put and set.
    Put sets a new value, overwriting the last one. Get destructively
    retrieves the value and blocks if none is set yet.
    """
    _sentinel = object()

    def __init__(self, value=_sentinel):
        self._cond = Condition(Lock())
        self._value = value

    def get(self):
        """
        Retrieves the value, blocking if its not set yet. This operation is destructive.
        """
        with self._cond:
            value = self._value
            if value is self._sentinel:
                self._cond.wait()
                value = self._value
            self._value = self._sentinel
            return value

    def set(self, value):
        """
        Sets a new value, overwriting the last one.
        """
        with self._cond:
            self._value = value
            self._cond.notify_all()


def process_one_frame(body_pix, input_tensor, background=None, *,
                      segmentation_threshold=0.6, internal_resolution=0.5):
    mask = body_pix.segmentize(
        input_tensor,
        segmentation_threshold=segmentation_threshold,
        internal_resolution=internal_resolution)
    dilated_mask = cv2.dilate(mask.numpy(), (8, 8), iterations=1)
    blurred_mask = cv2.blur(dilated_mask, (8, 8))
    final_mask = tf.reshape(blurred_mask, (input_tensor.shape[0], input_tensor.shape[1], 1))
    inverted_mask = 1 - final_mask
    if background is None:
        background = cv2.blur(input_tensor.numpy(), (64, 64))
    return background * inverted_mask + tf.dtypes.cast(input_tensor, tf.float32) * final_mask


def main():
    body_pix = MobileNetV1.from_path(sys.argv[1])
    if sys.argv[4] == 'replace':
        background = tf.image.resize(tf.image.decode_image(tf.io.read_file(sys.argv[5])), (720, 1280))
    else:
        background = None
    output_ref = MRef()
    input_ref = MRef()
    with FrameGrabber(sys.argv[2]) as frame_grabber, FrameWriter(sys.argv[3]) as frame_writer:
        Thread(target=frame_writer.run, args=(output_ref, )).start()
        Thread(target=frame_grabber.run, args=(input_ref, )).start()
        while True:
            img = input_ref.get()
            started = time.monotonic()
            result = process_one_frame(body_pix, img, background)
            output_ref.set(result)
            print(time.monotonic() - started)


main()
