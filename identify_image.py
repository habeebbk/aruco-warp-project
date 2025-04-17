import cv2
import numpy as np
import json
import os
import time

# Load marker-id map
with open('known_markers.json', 'r') as f:
    marker_id_map = json.load(f)

# Reverse lookup: ID -> image name
id_to_name = {}
for name, ids in marker_id_map.items():
    for i in ids:
        id_to_name[i] = name

# ArUco setup
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

cap = cv2.VideoCapture(0)

print("📷 Show an image to the camera. Press 'q' to quit.")

processed_images = set()
last_detected = None
last_update_time = 0
cooldown_start_time = 0
COOLDOWN_DURATION = 1.0  # 1 second

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # Skip detection if we're in cooldown
    if current_time - cooldown_start_time < COOLDOWN_DURATION:
        cv2.imshow("Live Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        detected_ids = ids.flatten().tolist()
        matches = {}

        for img_name, marker_ids in marker_id_map.items():
            common = set(marker_ids).intersection(detected_ids)
            if len(common) >= 4:
                matches[img_name] = len(common)

        if matches:
            best_match = max(matches, key=matches.get)

            if best_match not in processed_images:
                print(f"🔍 Detected: {best_match}")
                last_detected = best_match
                last_update_time = current_time

                id_to_corner = {i: c[0] for i, c in zip(ids.flatten(), corners)}
                ordered_ids = marker_id_map[best_match]

                try:
                    ordered_points = np.array([
                        id_to_corner[ordered_ids[0]][0],
                        id_to_corner[ordered_ids[1]][1],
                        id_to_corner[ordered_ids[2]][3],
                        id_to_corner[ordered_ids[3]][2],
                    ], dtype="float32")
                except KeyError:
                    print("⚠️ Some expected marker IDs not detected. Skipping.")
                    continue

                original_path = os.path.join("output", best_match)
                original_img = cv2.imread(original_path)

                if original_img is None:
                    print(f"⚠️ Original image not found: {original_path}")
                    continue

                padding_top = 150
                padding_bottom = 150
                padding_left = 150
                padding_right = 150

                h, w = original_img.shape[:2]
                padded_w = w + padding_left + padding_right
                padded_h = h + padding_top + padding_bottom

                dst_points = np.array([
                    [padding_left, padding_top],
                    [padding_left + w - 1, padding_top],
                    [padding_left, padding_top + h - 1],
                    [padding_left + w - 1, padding_top + h - 1]
                ], dtype="float32")

                M = cv2.getPerspectiveTransform(ordered_points, dst_points)
                warped = cv2.warpPerspective(frame, M, (padded_w, padded_h))

                # Save with name_uid format
                uid = time.strftime("%Y%m%d%H%M%S")
                filename = f"{os.path.splitext(best_match)[0]}_{uid}.jpg"
                save_path = os.path.join("output", filename)
                cv2.imwrite(save_path, warped)
                print(f"✅ Warped & saved: {save_path}")

                processed_images.add(best_match)
                cooldown_start_time = time.time()  # 🔥 Start cooldown after save

                cv2.putText(warped, f"Warped: {best_match}", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Warped Output", warped)

    else:
        # No detection case
        if current_time - last_update_time > 1:
            processed_images.clear()
            last_detected = None

    if corners:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    cv2.imshow("Live Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
