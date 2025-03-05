
import os
import csv

def logging_csv(number, mode, landmark_list):
    if mode == 0:
        return

    # Updated keypoint.csv directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "keypoint_classifier", "keypoint.csv")

    # ensure the directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Check if the file already exists
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

    # Open the file and write data
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)


        if not file_exists:
            header = ["label"] + [f"x{i}, y{i}" for i in range(len(landmark_list) // 2)]
            writer.writerow(header)

        # Ensure `number` is not -1
        if number != -1:
            writer.writerow([number, *landmark_list])
            print(f"✅ Data written: Class {number}, File {csv_path}")
        else:
            print("⚠️ `number` is still -1, data not written!")

