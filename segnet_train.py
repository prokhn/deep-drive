import os
import gc
import cv2
import tqdm
import random
import numpy as np
from skimage.io import imsave
from multiprocessing import Pool, Process, Queue

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

from segnet import SegnetBuilder
from keras.callbacks import EarlyStopping, ModelCheckpoint

# \ Globals ============
PIC_H, PIC_W = 672, 832
LABELS_NUMBER = 12
PROCESSES = 30


def get_filenames(datapath='lane_marking_examples'):
    filenames = []
    for top, dirs, files in os.walk(datapath):
        filenames.extend([os.path.join(top, _file) for _file in files])
    filenames.sort()

    x_paths = [x for x in filenames if not x.endswith('bin.png')]
    y_paths = [x for x in filenames if x.endswith('bin.png')]

    return x_paths, y_paths


def load_image(filepath, resize=True, pic_h=int(2710 / 4), pic_w=int(3384 / 4)):
    if resize:
        img = cv2.imread(filepath)
    else:
        img = cv2.resize(cv2.imread(y), (pic_h, pic_w), interpolation=cv2.INTER_NEAREST)

    return filepath, img


def load_pair(task_queue, out_queue):
    # Height and width to resize images
    # Defaults: 2710, 3384
    # Coef = 4: 677, 846 [OLD]
    # Coef ~ 4: 672, 832 [Divisible by 32]

    pic_h = 672
    pic_w = 832

    # Start: Labels conversion dict ===========================
    labels = {
        (0, 0, 0):       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        (8, 35, 142):    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        (43, 173, 180):  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        (153, 102, 153): [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        (234, 168, 160): [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        (192, 0, 0):     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        (8, 32, 128):    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        (12, 51, 204):   [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        (70, 25, 100):   [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        (14, 57, 230):   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        (75, 47, 190):   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        (255, 255, 255): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}
    labels = dict([(k, np.array(v)) for k, v in labels.items()])
    # End:   Labels conversion dict ===========================

    while True:
        xfilepath, yfilepath = task_queue.get()

        if xfilepath == '':
            break

        img = cv2.resize(cv2.imread(xfilepath), (pic_w, pic_h), interpolation=cv2.INTER_NEAREST)
        lbl = cv2.resize(cv2.imread(yfilepath), (pic_w, pic_h), interpolation=cv2.INTER_NEAREST)

        mask = np.zeros((pic_h, pic_w, len(labels)))
        for rgb, categorical_lbl in labels.items():
            mask[(lbl == rgb).all(2)] = categorical_lbl

        out_queue.put((img, mask))


def predict_to_label(predictions):
    labels = {
        0:  (0, 0, 0),
        1:  (8, 35, 142),
        2:  (43, 173, 180),
        3:  (153, 102, 153),
        4:  (234, 168, 160),
        5:  (192, 0, 0),
        6:  (8, 32, 128),
        7:  (12, 51, 204),
        8:  (70, 25, 100),
        9:  (14, 57, 230),
        10: (75, 47, 190),
        11: (255, 255, 255)}

    # masks = np.zeros((len(predictions), PIC_H, PIC_W, 3))
    masks = []
    for ind, pred in enumerate(predictions):
        pred = pred.reshape(PIC_H, PIC_W, 12)
        pred = np.apply_along_axis(lambda x: np.argmax(x), axis=2, arr=pred)
        h_pred = np.zeros((PIC_H, PIC_W, 3), dtype=np.uint8)
        for argmax, rgb in labels.items():
            h_pred[pred == argmax] = rgb

        masks.append(h_pred)

    return masks


def prepare_for_train(model_name=''):
    if model_name == '':
        model_name = f'Segnet {str(random.randint(0, 100)).zfill(3)}'

    segnet = SegnetBuilder.build(model_name, PIC_W, PIC_H, 3, LABELS_NUMBER)

    segnet.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

    return model_name, segnet


def train_on_batch(sup_epoch: int, model_name: str, segnet: SegnetBuilder.build, load_from: int, load_to: int):
    x_paths, y_paths = get_filenames()
    # print(f'Got {x_paths.__len__()} xdata filenames and {y_paths.__len__()} ydata filenames')

    load_total = load_to - load_from

    q_tasks = Queue()
    q_outs = Queue()

    for i in range(load_from, load_to):
        q_tasks.put((x_paths[i], y_paths[i]))
    for i in range(PROCESSES):
        q_tasks.put(('', ''))

    # Img and masks arrays
    x_data = np.zeros((load_total, PIC_H, PIC_W, 3), dtype=np.uint8)
    y_data = np.zeros((load_total, PIC_H, PIC_W, 12), dtype=np.float64)

    processes = [Process(target=load_pair, args=(q_tasks, q_outs,)) for _ in range(PROCESSES)]
    for p in processes:
        p.start()

    for i in tqdm.tqdm(range(load_total)):
        img, mask = q_outs.get()
        x_data[i] = img
        y_data[i] = mask

    print('Converting data...')
    x_data = x_data / 255.
    # print('Converting y_data... [reshaping]')
    y_data = y_data.reshape(len(y_data), PIC_H * PIC_W, LABELS_NUMBER)

    for p in processes:
        p.join()

    print(f'Data converted, starting {model_name} training at {sup_epoch} epoch')

    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001,
                               patience=3, verbose=1, mode='min')  # mode='auto' for val_acc
    checkpoint = ModelCheckpoint(f'models/{model_name} s[{sup_epoch}].hdf5',
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='auto')

    callbacks = [early_stop, checkpoint]

    segnet.fit(x_data, y_data, batch_size=1, epochs=1,
               verbose=1, validation_split=0.2, callbacks=callbacks)

    del x_data
    del y_data
    del q_tasks
    del q_outs
    del processes

    return segnet


def check_prediction(pred_index=999):
    print('Making predictions...')
    predictions = segnet.predict(np.expand_dims(x_data[pred_index], axis=0))

    print('Converting pred to human mask...')
    human_pred = predict_to_label(predictions)[0]
    print('Converting mask to human mask...')
    human_mask = predict_to_label([y_data[pred_index]])[0]

    print('Saving result...')
    imsave('img.png', np.concatenate((human_mask, human_pred), axis=0))
    print('Goto-vo')


if __name__ == '__main__':
    # Total 10464 images in dataset
    # 12 mini batches with 872 images in each
    # 2 epoch per each batch - 24 epochs total
    epochs_number = 2
    mini_batches = [(0, 872), (872, 1744), (1744, 2616), (2616, 3488),
                    (3488, 4360), (4360, 5232), (5232, 6104), (6104, 6976),
                    (6976, 7848), (7848, 8720), (8720, 9592), (9592, 10464 + 1)]

    model_name, segnet = prepare_for_train()

    print('='*50)
    print(f'Segnet "{model_name}" created and compiled'.center(50))
    print('='*50)

    for epoch in range(1, epochs_number + 1):
        print('=' * 50)
        print(f'Batch epoch {epoch} started')
        for batch_start, batch_end in mini_batches:
            print('-' * 50)
            print(f'Training on batch ({batch_start}, {batch_end})')
            segnet = train_on_batch(epoch, model_name, segnet, batch_start, batch_end)

            print('-' * 50)
            print(f'Training on batch ({batch_start}, {batch_end}) ended, clearing garbage')
            gc.collect()
            print('Garbage collected')

    print('Training ended without errors')
