import cv2
import numpy as np


def datapre(data):
    data = data[:, :410, :]
    data = cv2.resize(data, (64, 64))
    data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    _, data = cv2.threshold(data, 1, 255, cv2.THRESH_BINARY)
    '''
    cv2.namedWindow("qqq")
    cv2.imshow("qqq", data)
    '''
    data = data.reshape(1, 64, 64)
    return data


if __name__ == '__main__':
    datapre()
