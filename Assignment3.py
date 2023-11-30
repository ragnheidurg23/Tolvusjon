import cv2
import numpy as np

def order_corners(points):
    # Raða þeim eftir x hnitum
    points = sorted(points, key=lambda p: p[0])

    # Skipti þeim upp eftir y hnitum
    upper_pair = sorted(points[:2], key=lambda p: p[1])
    lower_pair = sorted(points[2:], key=lambda p: p[1])

    ordered_corners = [upper_pair[0], upper_pair[1], lower_pair[1], lower_pair[0]]

    return ordered_corners


cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 9)
    edges = cv2.Canny(gray, 100, 200)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=40, maxLineGap=30)

    if lines is not None:
        # Teikna línurnar
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Finn intersections með cross product
        intersections = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                line1 = lines[i][0]
                line2 = lines[j][0]

                hom_line1 = np.cross([line1[0], line1[1], 1], [line1[2], line1[3], 1])
                hom_line2 = np.cross([line2[0], line2[1], 1], [line2[2], line2[3], 1])

                # Samsíða línur eða ekki?
                if hom_line1[2] == 0 or hom_line2[2] == 0:
                    continue

                # Reikna cross
                intersection = np.cross(hom_line1, hom_line2)
                px, py, pw = map(int, intersection)

                if pw != 0:
                    px, py = px // pw, py // pw

                    intersections.append((px, py))

        # Taka 4 vinsælustu intersection
        if len(intersections) >= 4:
            intersection_counts = np.array(intersections)
            intersection_counts = np.unique(intersection_counts, axis=0, return_counts=True)
            sorted_indices = np.argsort(-intersection_counts[1])[:8]
            most_common_intersections = intersection_counts[0][sorted_indices]

            # Passa að þetta sé ekki sami punkturinn tvisvar
            valid_intersections = []
            for new_intersection in most_common_intersections:
                # Passa að þetta sé inní rammanum
                if 0 <= new_intersection[0] < frame.shape[1] and 0 <= new_intersection[1] < frame.shape[0]:
                    is_close = False
                    for prev_intersection in valid_intersections:
                        distance = np.linalg.norm(new_intersection - prev_intersection)
                        if distance < 30:
                            is_close = True
                            break
                    if not is_close:
                        valid_intersections.append(new_intersection)
                    if len(valid_intersections) == 4:
                        break

            for intersection in valid_intersections:
                cv2.circle(frame, tuple(intersection), 10, (0, 255, 0), -1)

            if len(valid_intersections) == 4:
                ordered_intersections = order_corners(valid_intersections)
                # Homography
                pts1 = np.float32(ordered_intersections)
                pts2 = np.float32([(0, 0), (300-1, 0), (300-1, 200-1), (0, 200-1)])
                matrix, x = cv2.findHomography(pts1, pts2)

                # Perspective transformation
                result = cv2.warpPerspective(frame, matrix, (300, 200))

                cv2.imshow("Rectified Image", result)

    cv2.imshow("Video with Lines", frame)
    cv2.imshow('Edges', edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
