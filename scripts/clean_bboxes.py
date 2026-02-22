import os
import argparse
from collections import Counter

import pandas as pd
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description='Clean bounding boxes in CSV by clamping to image bounds')
    p.add_argument('--csv', default='data/raw/Train.csv', help='Input CSV path')
    p.add_argument('--img-dir', default='data/raw/images', help='Images directory')
    p.add_argument('--out-csv', default='data/processed/Train_clean.csv', help='Output cleaned CSV')
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.csv)

    # ensure numeric bbox columns
    for c in ('xmin', 'ymin', 'xmax', 'ymax'):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            df[c] = pd.NA

    total_before = len(df)
    fixed = 0
    dropped = 0
    clean_rows = []

    # process per-image to avoid reopening images repeatedly
    grouped = df.groupby('Image_ID')
    for img_id, group in tqdm(grouped, total=len(grouped), desc='Images'):
        img_path = os.path.join(args.img_dir, img_id)
        if not os.path.exists(img_path):
            dropped += len(group)
            continue

        import cv2
        img = cv2.imread(img_path)
        if img is None:
            dropped += len(group)
            continue
        h, w = img.shape[:2]

        for _, row in group.iterrows():
            xmin = row['xmin']
            ymin = row['ymin']
            xmax = row['xmax']
            ymax = row['ymax']

            # drop rows with NaNs in coords
            if pd.isna(xmin) or pd.isna(ymin) or pd.isna(xmax) or pd.isna(ymax):
                dropped += 1
                continue

            # Fix ordering: if invalid, drop
            if xmin >= xmax or ymin >= ymax:
                dropped += 1
                continue

            # Clamp to image bounds (keep floats)
            new_xmin = max(0.0, min(float(w) - 1.0, float(xmin)))
            new_ymin = max(0.0, min(float(h) - 1.0, float(ymin)))
            new_xmax = max(1.0, min(float(w), float(xmax)))
            new_ymax = max(1.0, min(float(h), float(ymax)))

            if new_xmin != float(xmin) or new_ymin != float(ymin) or new_xmax != float(xmax) or new_ymax != float(ymax):
                fixed += 1

            # Re-check validity after clamp
            if new_xmin >= new_xmax or new_ymin >= new_ymax:
                dropped += 1
                continue

            row['xmin'] = new_xmin
            row['ymin'] = new_ymin
            row['xmax'] = new_xmax
            row['ymax'] = new_ymax

            clean_rows.append(row)

    clean_df = pd.DataFrame(clean_rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    clean_df.to_csv(args.out_csv, index=False)

    print('✅ Cleaning complete')
    print(f'Total rows before: {total_before}')
    print(f'Rows after cleaning: {len(clean_df)}')
    print(f'Fixed boxes: {fixed}')
    print(f'Dropped boxes: {dropped}')
    print(f'Saved to: {args.out_csv}')


if __name__ == '__main__':
    main()
