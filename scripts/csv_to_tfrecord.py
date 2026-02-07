import os
import argparse
from collections import defaultdict

import cv2
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default='data/processed/Train_clean.csv')
    p.add_argument('--img-dir', default='data/raw/images')
    p.add_argument('--out-dir', default='data/processed/tfrecords')
    p.add_argument('--val-size', type=float, default=0.2)
    return p.parse_args()


def _bytes(v):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))


def _float(v):
    return tf.train.Feature(float_list=tf.train.FloatList(value=v))


def _int(v):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=v))


def create_example(image_id, rows, img_dir, class_map):
    img_path = os.path.join(img_dir, image_id)
    with tf.io.gfile.GFile(img_path, 'rb') as fid:
        encoded = fid.read()
    img = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('Could not decode image: ' + img_path)
    h, w = img.shape[:2]

    xmins, ymins, xmaxs, ymaxs, labels = [], [], [], [], []
    for _, r in rows.iterrows():
        xmins.append(float(r['xmin']) / w)
        xmaxs.append(float(r['xmax']) / w)
        ymins.append(float(r['ymin']) / h)
        ymaxs.append(float(r['ymax']) / h)
        labels.append(int(class_map[r['class']]))

    feature = {
        'image/encoded': _bytes(encoded),
        'image/filename': _bytes(image_id.encode()),
        'image/height': _int([h]),
        'image/width': _int([w]),
        'image/object/bbox/xmin': _float(xmins),
        'image/object/bbox/xmax': _float(xmaxs),
        'image/object/bbox/ymin': _float(ymins),
        'image/object/bbox/ymax': _float(ymaxs),
        'image/object/class/label': _int(labels),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_tfrecord(ids, groups, out_path, img_dir, class_map):
    with tf.io.TFRecordWriter(out_path) as writer:
        for img_id in tqdm(ids, desc=os.path.basename(out_path)):
            try:
                ex = create_example(img_id, groups.get_group(img_id), img_dir, class_map)
                writer.write(ex.SerializeToString())
            except Exception as e:
                print('Skipping', img_id, e)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    classes = sorted(df['class'].unique())
    class_map = {c: i + 1 for i, c in enumerate(classes)}

    # save label map
    with open('data/processed/label_map.txt', 'w') as f:
        for k, v in class_map.items():
            f.write(f"{k}: {v}\n")
    print('Saved label_map.txt')

    groups = df.groupby('Image_ID')
    image_ids = list(groups.groups.keys())
    train_ids, val_ids = train_test_split(image_ids, test_size=args.val_size, random_state=42)

    write_tfrecord(train_ids, groups, os.path.join(args.out_dir, 'train.tfrecord'), args.img_dir, class_map)
    write_tfrecord(val_ids, groups, os.path.join(args.out_dir, 'val.tfrecord'), args.img_dir, class_map)

    print('✅ TFRecords created')


if __name__ == '__main__':
    import numpy as np
    main()
