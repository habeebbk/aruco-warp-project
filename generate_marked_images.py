import cv2
import numpy as np
import os
import json

# ==== CONFIGURATION ====
input_folder = 'input'
output_folder = 'output'
aruco_dict_type = cv2.aruco.DICT_4X4_50
aruco_size = 100
padding = 50
start_marker_id = 0  # Will be incremented
marker_id_map = {}
# ========================

os.makedirs(output_folder, exist_ok=True)
aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)

for img_name in os.listdir(input_folder):
    if not img_name.lower().endswith(('.jpg', '.png')):
        continue

    input_path = os.path.join(input_folder, img_name)
    img = cv2.imread(input_path)
    if img is None:
        print(f"❌ Couldn't read {img_name}")
        continue

    h, w = img.shape[:2]
    new_w = w + 2 * (aruco_size + padding)
    new_h = h + 2 * (aruco_size + padding)
    canvas = np.ones((new_h, new_w, 3), dtype=np.uint8) * 255
    x_offset = aruco_size + padding
    y_offset = aruco_size + padding
    canvas[y_offset:y_offset + h, x_offset:x_offset + w] = img

    # Assign 4 marker IDs
    marker_ids = list(range(start_marker_id, start_marker_id + 4))
    marker_id_map[img_name] = marker_ids
    start_marker_id += 4

    positions = [
        (padding, padding),  # top-left
        (new_w - padding - aruco_size, padding),  # top-right
        (padding, new_h - padding - aruco_size),  # bottom-left
        (new_w - padding - aruco_size, new_h - padding - aruco_size)  # bottom-right
    ]

    for id, pos in zip(marker_ids, positions):
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, id, aruco_size)
        marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        x, y = pos
        canvas[y:y+aruco_size, x:x+aruco_size] = marker_bgr

    output_path = os.path.join(output_folder, img_name)
    cv2.imwrite(output_path, canvas)
    print(f"✅ Saved: {output_path}")

# Save ID mapping
with open('known_markers.json', 'w') as f:
    json.dump(marker_id_map, f, indent=2)

print("📝 Marker-ID map saved to known_markers.json")
