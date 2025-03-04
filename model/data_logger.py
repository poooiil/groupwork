
import os
import csv

def logging_csv(number, mode, landmark_list):
    if mode == 0:
        return

    # **修改后的 keypoint.csv 目录**
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "keypoint_classifier", "keypoint.csv")

    # **确保目录存在**
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # **检查文件是否已存在**
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

    # **打开文件写入数据**
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)

        # **如果是新文件，写入标题行**
        if not file_exists:
            header = ["label"] + [f"x{i}, y{i}" for i in range(len(landmark_list) // 2)]
            writer.writerow(header)

        # **确保 `number` 不是 -1**
        if number != -1:
            writer.writerow([number, *landmark_list])
            print(f"✅ 已写入数据: 类别 {number}, 文件 {csv_path}")
        else:
            print("⚠️ `number` 仍然是 -1，未写入数据！")

