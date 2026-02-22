#!/usr/bin/env python3
import os
import shutil
import argparse
from collections import defaultdict

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default='data/processed/Train_clean.csv')
    p.add_argument('--img-dir', default='data/raw/images')
    p.add_argument('--out-dir', default='data/processed/yolo')
    p.add_argument('--val-size', type=float, default=0.2)
    return p.parse_args()


def load_label_map(path='data/processed/label_map.txt'):
    m = {}
    if not os.path.exists(path):
        return m
    with open(path) as f:
        for line in f:
            line=line.strip()
            if not line: continue
            if ':' in line:
                k,v=line.split(':',1)
                m[k.strip()]=int(v.strip())
    return m


def ensure_dirs(base):
    for split in ('train','val'):
        os.makedirs(os.path.join(base, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(base, split, 'labels'), exist_ok=True)


def to_yolo_box(xmin,ymin,xmax,ymax,w,h):
    # convert to x_center,y_center,width,height normalized
    x_center = (xmin + xmax) / 2.0 / w
    y_center = (ymin + ymax) / 2.0 / h
    bw = (xmax - xmin) / w
    bh = (ymax - ymin) / h
    return x_center, y_center, bw, bh


def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    label_map = load_label_map()
    ensure_dirs(args.out_dir)

    groups = df.groupby('Image_ID')
    image_ids = list(groups.groups.keys())
    train_ids, val_ids = train_test_split(image_ids, test_size=args.val_size, random_state=42)

    for split, ids in (('train', train_ids), ('val', val_ids)):
        for img_id in ids:
            img_src = os.path.join(args.img_dir, img_id)
            dst_img = os.path.join(args.out_dir, split, 'images', img_id)
            if not os.path.exists(img_src):
                continue
            shutil.copy2(img_src, dst_img)
            rows = groups.get_group(img_id)
            # read image size via cv2 only if needed; prefer PIL-less approach: use OpenCV
            import cv2
            img = cv2.imread(img_src)
            if img is None:
                continue
            h,w = img.shape[:2]
            label_lines = []
            for _, r in rows.iterrows():
                cls = r['class']
                if cls not in label_map:
                    continue
                cid = label_map[cls] - 1  # YOLO class ids 0-based
                xmin = float(r['xmin'])
                ymin = float(r['ymin'])
                xmax = float(r['xmax'])
                ymax = float(r['ymax'])
                xc,yc,bw,bh = to_yolo_box(xmin,ymin,xmax,ymax,w,h)
                label_lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
            lbl_path = os.path.join(args.out_dir, split, 'labels', os.path.splitext(img_id)[0]+'.txt')
            with open(lbl_path, 'w') as f:
                f.writelines(label_lines)

    print('Created YOLO-format dataset at', args.out_dir)


if __name__ == '__main__':
    main()
