import struct
import subprocess
import sys
from queue import Queue
from threading import Thread

import cv2
import numpy as np
import tensorflow as tf


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
        # Normalize the pixels [0, 255] to be between [-1, 1].
        normalized = input / 127.5 - 1.0
        batch = tf.expand_dims(normalized, 0)
        return self._predict_function(batch)


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

    def run(self, queue):
        size_struct = struct.Struct('<I')
        while True:
            header = self._ffmpeg.stdout.read(6)
            if header[0:2] != b"BM":
                raise RuntimeError("Expected Bitmap header not found")
            (size, ) = size_struct.unpack_from(header, 2)
            bmp = tf.image.decode_bmp(header + self._ffmpeg.stdout.read(size - 6))
            queue.put(bmp)


class FrameWriter:
    def __init__(self, device):
        self.device = device

    def __enter__(self):
        self._ffmpeg = subprocess.Popen(
            ['ffmpeg', '-i', '-', '-pix_fmt', 'yuv420p', '-f', 'v4l2',
             '-video_size', 'hd720', '-r', '30', self.device],
            stdin=subprocess.PIPE)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._ffmpeg.kill()

    def run(self, queue):
        while True:
            tensor = queue.get()
            as_uint8 = tf.dtypes.cast(tensor, tf.uint8)
            self._ffmpeg.stdin.write(tf.image.encode_png(as_uint8, compression=0).numpy())
            self._ffmpeg.stdin.flush()


def main():
    body_pix = BodyPix.from_path(sys.argv[2])
    background = tf.image.resize(tf.image.decode_image(tf.io.read_file(sys.argv[1])), (720, 1280))
    output_queue = Queue()
    input_queue = Queue()
    import time
    with FrameGrabber(sys.argv[3]) as frame_grabber, FrameWriter(sys.argv[4]) as frame_writer:
        Thread(target=frame_writer.run, args=(output_queue, )).start()
        Thread(target=frame_grabber.run, args=(input_queue, )).start()
        while True:
            while input_queue.qsize() > 2:
                # Drop frames
                input_queue.get()
            img = input_queue.get()
            started = time.monotonic()
            mask = body_pix.segmentize(img, segmentation_threshold=0.5, internal_resolution=0.75)
            dilated_mask = cv2.dilate(mask.numpy(), np.ones((32, 32), np.uint8) , iterations=1)
            eroded_mask = cv2.erode(dilated_mask, np.ones((16, 16), np.uint8) , iterations=1)
            final_mask = tf.reshape(eroded_mask, (img.shape[0], img.shape[1], 1))
            inverted_mask = 1 - final_mask
            result = background * inverted_mask + tf.dtypes.cast(img, tf.float32) * final_mask
            output_queue.put(result)
            print(time.monotonic() - started)


main()
