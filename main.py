import json
import os
import time
from queue import Queue
from threading import Thread

import cv2
import numpy
import torch

root = '/mnt/nas/Fotos/'
extensions = ['jpg', 'jpeg', 'png']

file_tree_limit = None

file_read_queue = Queue(maxsize=-1)
file_process_queue = Queue(maxsize=100)
file_data_queue = Queue(maxsize=100)

detections_counter = 0


def load_model(device_name: str):
    model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'yolov7.pt')
    device = torch.device(device_name)
    model.to(device)
    return model


def thread_file_reader():
    while True:
        filepath = file_read_queue.get()

        image = cv2.imread(filepath)
        if image is None:
            continue

        file_process_queue.put((filepath, image))


def thread_file_processor(device_name: str):
    global detections_counter

    print(f'Loading model on device {device_name}...')
    model = load_model(device_name)

    while True:
        try:
            filepath, image = file_process_queue.get()

            results = process_image(model, filepath, image)
            detections_counter += 1

            file_data_queue.put(results)
        except Exception as e:
            print(f'IGNORING EXCEPTION IN thread_file_processor("{device_name}"): {e}')


def thread_data_store():
    file = open('output.txt', 'a')

    while True:
        results = file_data_queue.get()
        file.write(f'{json.dumps(results)}\n')


def process_image(model, filepath, image: numpy.ndarray):
    panda = model(image).pandas().xyxy[0]

    width, height = image.shape[:2]

    classifications = []
    for index, row in panda.iterrows():
        classifications.append({
            'name': row['name'],
            'confidence': row['confidence'],
            'x1': row['xmin'] / width,
            'y1': row['ymin'] / height,
            'x2': row['xmax'] / width,
            'y2': row['ymax'] / height,
        })

    return {
        'filepath': filepath,
        'width': width,
        'height': height,
        'classifications': classifications,
    }

def load_already_processed_filepaths():
    filepaths = set()
    with open('output.txt') as file:
        for line in file:
            data = json.loads(line)
            filepaths.add(data['filepath'])
    return filepaths


def main():
    Thread(target=thread_file_processor, args=['cuda:0']).start()
    Thread(target=thread_file_processor, args=['cuda:1']).start()

    already_processed_filepaths = load_already_processed_filepaths()

    print('Generating filetree...')
    total_files = 0
    skipped_files = 0
    for (dirpath, dirnames, filenames) in os.walk(root):
        for filename in filenames:
            filepath = f'{dirpath}/{filename}'
            extension = filename.split('.')[-1].lower()

            if filepath in already_processed_filepaths:
                skipped_files += 1
                continue

            if extension in extensions:
                file_read_queue.put(filepath)
                total_files += 1

            if file_tree_limit is not None and total_files >= file_tree_limit:
                print('REACHED FILE TREE LIMIT!')
                break

        if file_tree_limit is not None and total_files >= file_tree_limit:
            break

    print(f'Done. Total amount of files to be processed is {total_files}. Skipped {skipped_files} which were already processed.')

    for i in range(0, 20):
        Thread(target=thread_file_reader).start()

    Thread(target=thread_data_store).start()

    last_detections_counter = 0
    while True:
        time.sleep(1.0)

        print(f'detections/s={detections_counter - last_detections_counter} '
              f'{detections_counter=} '
              f'percent_completed={detections_counter / total_files * 100:.4f}% '
              f'{file_read_queue.qsize()=} '
              f'{file_process_queue.qsize()=}')

        last_detections_counter = detections_counter


if __name__ == '__main__':
    main()
