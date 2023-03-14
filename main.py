import time

import cv2
import numpy as np

from detector.detector import Detector
from detector.letter_model import LetterModel
from figures.generate import get_figure_sample, load_classes
from figures.util import load_map

if __name__ == "__main__":
    map_img = load_map("./assets/suasorto.tif")

    d = Detector("assets/yolo_weights.pt")
    lm = LetterModel("assets/model_weights.pth")

    time_avg = 0
    for i in range(5):
        start = time.time()
        img, cxywh = get_figure_sample(load_classes(), map_img)
        results = d.detect_with_letters(img, lm)
        time_avg += (time.time() - start)
        for r in results:
            print()
            print("number of figures detected: ", len(results))
            print("box:", r["box"])
            print("conf:", r["conf"])
            print("cls:", r["cls"])
            print("label:", r["label"])
            if "letter" in r:
                print("letter:", r["letter"])

            cv2.imshow("test", np.asarray(r["im"]))
            cv2.waitKey(2000)

    print("time average: ", time_avg / 100)
