# Hand Gesture Recognition (ML) Model for MysticGesturesGame

## Dependencies
- Python >= 3.9.6
- OpenCV >= 4.11.0.86
- PyTorch >= 2.6.0
- MediaPipe >= 0.10.21
- Pandas >= 2.2.3
- Scikit-learn == 1.5.1

## Usage

Argument | Type | Default Value | Description
-|-|-|-
`--device` | `int` | `0` | Specifies the device to use (e.g., camera ID).
`--width` | `int` | `960` | Sets the capture width.
`--height`| `int`   | `540`| Sets the capture height.
`--use_static_image_mode` | `bool`  | `False` | Enables or disables static image mode.
`--min_detection_confidence` | `float` | `0.7` | The minimum confidence threshold for detection.
`--min_tracking_confidence` | `float` | `0.5` | The minimum confidence threshold for tracking.
`--server_address` | `str` | `'127.0.0.1'` | The UDP server address to connect to.
`--server_port` | `int` | `65432` | The UDP server port to connect to.
`--stability_threshold` | `int` | `2000` | Sets the stability threshold in milliseconds for prediction.


```bash
# Example usage (UV)
$ uv run python model/main.py --device=1 --width=1280 --height=720 --server_address='192.168.1.100' --server_port=12345
```

## Output

ID | Label | Gesture
-|-|-
`0` | `moves` | Move
`1` | `fire` | Fire
`2` | `explode` | Explode
`3` | `wind` | Wind
`4` | `shrink` | Shrink
