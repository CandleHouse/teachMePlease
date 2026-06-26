"""
手势识别 —— 数手指，输出阿拉伯数字
====================================
完整流程：
1. 调用电脑摄像头，逐帧采集画面
2. MediaPipe HandLandmarker 提取每只手 21 个关键点
3. 用「指尖 vs 指根」的相对位置判断每根手指伸/弯
4. 统计伸直的手指数量 → 阿拉伯数字 0~5
5. 在画面上用方框标记手的位置，并显示识别结果

操作：
    q  —— 退出

依赖：
    pip install mediapipe opencv-python numpy
    模型文件 hand_landmarker.task 需与本脚本放在同一目录
    （首次缺失时脚本会自动从 Google 官方下载一次）。
"""

import os
import sys
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


# ════════════════════════════════════════════════
# 1. 模型文件准备
# ════════════════════════════════════════════════

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")


def ensure_model():
    """确保 hand_landmarker.task 存在；不存在则下载一次。"""
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
        return
    print("首次运行：正在下载手部关键点模型 hand_landmarker.task ...", flush=True)
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("模型下载完成。", flush=True)
    except Exception as e:
        print(f"模型自动下载失败：{e}", file=sys.stderr)
        print(f"请手动下载 {MODEL_URL} 并放到 {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)


# ════════════════════════════════════════════════
# 2. 手指状态判定
# ════════════════════════════════════════════════

# MediaPipe HandLandmarker 标定每只手 21 个关键点，编号固定：
#
#    4
#    |
#    3        8   12  16  20        ← 5 个指尖
#    |        |   |   |   |
#    2        7   11  15  19        ← PIP 关节（第二指节根）
#    |        |   |   |   |
#    1        6   10  14  18        ← MCP 关节（掌指关节）
#             |   |   |   |
#             5   9   13  17        ← 五指的掌指关节连成掌
#    0                              ← 手腕
#
# 判断一根手指「是否伸直」的方法：
#   · 食指~小指：指尖比 PIP 关节更靠「上」即伸直
#   · 大拇指：靠水平方向比较，指尖比 IP 关节更靠「外」即伸直

FINGER_TIPS = [4, 8, 12, 16, 20]      # 大拇指、食指、中指、无名指、小指
FINGER_PIPS = [3, 6, 10, 14, 18]      # 对应的上一级关节


def count_fingers(landmarks, handedness_label, image_width, image_height):
    """
    根据一只手的 21 个关键点，判断有几根手指伸直。

    Parameters
    ----------
    landmarks : list[NormalizedLandmark]
        HandLandmarker 返回的归一化关键点，坐标范围 [0, 1]。
    handedness_label : str
        "Left" 或 "Right"，决定大拇指的左右判定方向。
    image_width, image_height : int
        画面像素尺寸，用于把归一化坐标换算成像素。

    Returns
    -------
    count : int
        伸直的手指数量（0~5）。
    states : list[bool]
        每根手指是否伸直，顺序 = [拇指, 食指, 中指, 无名指, 小指]。
    """
    pts = [(lm.x * image_width, lm.y * image_height) for lm in landmarks]

    states = []

    # —— 大拇指（横向比较）——
    # 镜像画面里，右手的大拇指指尖在更靠右时为伸直，左手相反。
    thumb_extended = (pts[4][0] > pts[3][0]) if handedness_label == "Right" \
        else (pts[4][0] < pts[3][0])
    states.append(thumb_extended)

    # —— 其余四指（纵向比较：指尖 y 更小 = 更靠上 = 伸直）——
    for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        states.append(pts[tip][1] < pts[pip][1])

    return sum(states), states


# ════════════════════════════════════════════════
# 3. 画面绘制
# ════════════════════════════════════════════════

# 骨架连线（21 个关键点之间的连接关系）
HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS


def draw_result(image, landmarks, count, states, handedness_label):
    """
    在一帧画面上：
      · 画出 21 个关键点骨架
      · 用方框框出手所在区域
      · 在方框上方标注「手势 → 数字」与各指状态
    """
    h, w = image.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    # --- 方框：取所有关键点的包围盒，外扩一点余量 ---
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    pad = 25
    x1, x2 = max(0, min(xs) - pad), min(w, max(xs) + pad)
    y1, y2 = max(0, min(ys) - pad), min(h, max(ys) + pad)

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 128), 3)

    # --- 骨架连线 ---
    for conn in HAND_CONNECTIONS:
        cv2.line(image, pts[conn.start], pts[conn.end], (60, 180, 255), 2)
    for idx, (px, py) in enumerate(pts):
        color = (0, 255, 255) if idx in FINGER_TIPS else (255, 255, 255)
        radius = 6 if idx in FINGER_TIPS else 3
        cv2.circle(image, (px, py), radius, color, -1)

    # --- 文字标注 ---
    cv2.rectangle(image, (x1, y1 - 52), (x1 + 220, y1 - 6), (40, 40, 40), -1)
    cv2.putText(image, f"{handedness_label}:  {count}", (x1 + 8, y1 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 128), 2)

    # 屏幕中央的巨大数字，方便远看
    cv2.putText(image, str(count), (w // 2 - 60, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0, 255, 128), 6, cv2.LINE_AA)

    # 各指状态条：| 表示伸直，_ 表示弯曲
    finger_names = ["Thumb", "Index", "Mid", "Ring", "Pinky"]
    detail = "  ".join(f"{n[:2]}:{'|' if s else '_'}" for n, s in zip(finger_names, states))
    cv2.putText(image, detail, (x1, y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)


# ════════════════════════════════════════════════
# 4. 主循环
# ════════════════════════════════════════════════

def main():
    ensure_model()

    cap = cv2.VideoCapture(0)                       # 0 = 默认摄像头
    if not cap.isOpened():
        print("无法打开摄像头，请检查设备/权限。", file=sys.stderr)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("摄像头已打开。")
    print("  在镜头前伸出 0~5 根手指，画面会实时显示识别出的数字。")
    print("  按 q 退出。")

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    with vision.HandLandmarker.create_from_options(options) as landmarker:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # 镜像翻转，像照镜子一样符合直觉
            frame = cv2.flip(frame, 1)

            # HandLandmarker 需要 RGB 输入
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # VIDEO 模式需要单调递增的时间戳（毫秒）
            ts_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, ts_ms)

            if result.hand_landmarks:
                for lm, handed in zip(result.hand_landmarks, result.handedness):
                    # 镜像翻转后，MediaPipe 报的 Left/Right 要反过来
                    raw = handed[0].category_name
                    label = "Left" if raw == "Right" else "Right"
                    h, w = frame.shape[:2]
                    count, states = count_fingers(lm, label, w, h)
                    draw_result(frame, lm, count, states, label)
            else:
                cv2.putText(frame, "Show your hand to the camera",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (120, 120, 120), 2)

            cv2.imshow("Gesture Recognition  (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
