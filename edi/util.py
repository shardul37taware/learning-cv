import numpy as np
import colorsys

def get_limits(color):
    c = np.uint8([[color]])
    hsvC = colorsys.rgb_to_hsv(color[2]/255, color[1]/255, color[0]/255)

    lowerLimit = np.array([hsvC[0] * 179 - 10, 100, 100])
    upperLimit = np.array([hsvC[0] * 179 + 10, 255, 255])

    lowerLimit[0] = max(lowerLimit[0], 0)
    upperLimit[0] = min(upperLimit[0], 179)

    return lowerLimit.astype(int), upperLimit.astype(int)
