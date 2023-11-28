import cv2
import numpy as np
from skimage.measure import ransac, LineModelND

def fit_line_ransac(data, iterations=100, threshold=2):
    best_model = None
    best_inliers = []

    for _ in range(iterations):
        sample_indices = np.random.choice(len(data), 2, replace=False)
        sample_points = data[sample_indices]

        x1, y1 = sample_points[0]
        x2, y2 = sample_points[1]

        if x1 == x2 and y1 == y2:
            continue

        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        distances = np.abs(m * data[:, 0] - data[:, 1] + b) / np.sqrt(m**2 + 1)

        inliers = data[distances < threshold]

        if len(inliers) > len(best_inliers):
            best_model = (m, b)
            best_inliers = inliers[0]

    return best_model, best_inliers

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 130)

    edge_points = np.column_stack(np.where(edges > 0))

    # Þessi lína notar ransac functionið frá scikitlearn
    model, inliers = ransac(edge_points, LineModelND, min_samples=2, residual_threshold=2)
    # Þetta er mitt ransac sem virkar samt ekki jafn vel og functionið frá ransac
    # model, inliers = fit_line_ransac(edge_points, 2, 2)

    #Bý til línuna
    if model is not None:
        y1, x1 = edge_points[inliers].min(axis=0)
        y2, x2 = edge_points[inliers].max(axis=0)

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Video with Line", frame)
    cv2.imshow('Edges', edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()