import shutil
from pathlib import Path
from typing import List

import cv2
import numpy as np


def mask_to_polygon(mask: np.array, report=False) -> List[int]:
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []

    for point in max(contours, key=cv2.contourArea):
        polygons.append(point[0][0] / mask.shape[1])
        polygons.append(point[0][1] / mask.shape[0])

    if report:
        print(f"Number of points = {len(polygons)}")

    return polygons


def main():
    Path("data/processed/TrainDataset/images").mkdir(parents=True, exist_ok=True)
    Path("data/processed/TrainDataset/labels").mkdir(parents=True, exist_ok=True)
    Path("data/processed/TestDataset").mkdir(parents=True, exist_ok=True)
    Path("data/processed/TestDataset").mkdir(parents=True, exist_ok=True)

    # for image_path in Path("data/raw/TrainDataset/image/").glob("*.png"):
    #     shutil.copy(image_path, "data/processed/TrainDataset/images")
    #     mask_path = Path("data/raw/TrainDataset/mask/") / image_path.name
    #     mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    #     polygon = mask_to_polygon(mask)
    #
    #     with open("data/processed/TrainDataset/labels/" + image_path.stem + ".txt", "w") as f:
    #         f.write("0 ")
    #         f.write(" ".join([str(x) for x in polygon]))

    for folder in Path("data/raw/TestDataset").iterdir():
        for image_path in (folder / "images").glob("*.png"):

            mask_path = str(folder / "mask" / image_path.name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if mask is None:
                print(f"Mask not found for {image_path}")
                continue
            
            polygon = mask_to_polygon(mask)

            shutil.copy(image_path, "data/processed/TestDataset/images")
            with open("data/processed/TestDataset/labels/" + image_path.stem + ".txt", "w") as f:
                f.write("0 ")
                f.write(" ".join([str(x) for x in polygon]))


if __name__ == '__main__':
    main()
