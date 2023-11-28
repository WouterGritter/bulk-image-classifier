import json
import math
import os
import random
from queue import Queue
from threading import Thread

import cv2


def load_results(filename):
    results = []
    with open(filename) as file:
        for line in file:
            results.append(json.loads(line))

    return results


def find_image(results, threshold, required_classifications):
    if required_classifications is None or len(required_classifications) == 0:
        return random.choice(results)

    choices = []
    for result in results:
        classifications = [x['name'] for x in result['classifications'] if x['confidence'] > threshold]
        is_allowed = True
        for c in required_classifications:
            if c not in classifications:
                is_allowed = False
                break
            classifications.remove(c)

        if is_allowed:
            choices.append(result)


    return random.choice(choices)


def add_bounding_boxes(image, data):
    image = image.copy()
    height, width = image.shape[:2]

    for res in data['classifications']:
        x1 = math.floor(res['x1'] * width)
        y1 = math.floor(res['y1'] * height)
        x2 = math.floor(res['x2'] * width)
        y2 = math.floor(res['y2'] * height)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )

        cv2.putText(
            image,
            f'{res["name"]}-{res["confidence"] * 100:.1f}%',
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

    return image


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image, 1.0

    if width is None:
        ratio = height / float(h)
        dim = (int(w * ratio), height)
    else:
        ratio = width / float(w)
        dim = (width, int(h * ratio))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


image_queue = Queue(5)

def thread_load_images():
    all_results = load_results('output.txt')

    while True:
        print('Loading next image...')
        data = find_image(all_results, 0.9, ['person', 'bird'])
        image = cv2.imread(data['filepath'])
        image = image_resize(image, height=700)

        image_queue.put((image, data))

def main():
    Thread(target=thread_load_images).start()

    while True:
        image, data = image_queue.get()
        image_bb = add_bounding_boxes(image, data)

        cv2.imshow('image', image_bb)

        key_pressed = cv2.waitKey(0) & 0xFF
        if key_pressed == ord('q'):
            os._exit(0)
        elif key_pressed == ord('b'):
            cv2.imshow('image', image)
            cv2.waitKey(0)


if __name__ == '__main__':
    main()
