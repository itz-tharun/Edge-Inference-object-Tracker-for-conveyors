

import cv2


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("ZOOM:", cap.get(cv2.CAP_PROP_ZOOM))
print("PAN:", cap.get(cv2.CAP_PROP_PAN))
print("TILT:", cap.get(cv2.CAP_PROP_TILT))
print("BACKEND:", cap.getBackendName())