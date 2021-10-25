from xml.dom import minidom
import os
import cv2
import numpy as np


def svg2label(svgFileName, classNames=['Background', 'Wall']):
    # mask initialization
    dom = minidom.parse(svgFileName)
    width = dom.getElementsByTagName('width')[0].firstChild.data
    width = int(width)
    height = dom.getElementsByTagName('height')[0].firstChild.data
    height = int(height)
    maskLabel = np.zeros((height, width, 1), dtype="uint8")
    # draw class
    polygonList = dom.getElementsByTagName('polygon')
    for i in range(len(polygonList)):
        className = polygonList[i].getAttribute('class')
        if className == 'Wall':
            classID = classNames.index(className)
            points = polygonList[i].getAttribute('points')
            points = points.split()
            polygon_points_list = []
            for point in points:
                pointX, pointY = point.split(',')
                pointX = int(float(pointX))
                pointY = int(float(pointY))
                polygon_points_list.append([pointX, pointY])
            polygon_points_np = np.array(polygon_points_list, np.int32)
            # classID = 1
            cv2.fillConvexPoly(maskLabel, polygon_points_np, color=classID)
    return maskLabel


def mask2onehot(mask):
    onehot = mask
    return onehot