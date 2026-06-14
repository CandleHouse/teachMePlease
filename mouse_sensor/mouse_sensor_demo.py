"""
光学鼠标传感器模拟
==================
完整流程：
1. Simplex Noise fBm 生成表面纹理
2. 模拟传感器逐帧拍摄 18x18 图像
3. 频域相位相关法计算帧间位移
4. 累加位移得到鼠标移动轨迹
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ════════════════════════════════════════════════
# 1. 2D Simplex Noise
# ════════════════════════════════════════════════

class SimplexNoise2D:

    F2 = (math.sqrt(3.0) - 1.0) / 2.0
    G2 = (3.0 - math.sqrt(3.0)) / 6.0
    GRAD2 = np.array([
        (1, 1), (-1, 1), (1, -1), (-1, -1),
        (1, 0), (-1, 0), (0, 1), (0, -1),
    ], dtype=np.float64)

    def __init__(self, seed=42):
        rng = np.random.RandomState(seed)
        p = np.arange(256, dtype=np.int32)
        rng.shuffle(p)
        self.perm = np.concatenate([p, p]).astype(np.int32)
        self.perm_mod8 = self.perm % 8

    def generate(self, width, height, scale=1.0):
        """向量化生成整张 Simplex Noise 图"""
        ys, xs = np.mgrid[0:height, 0:width]
        xs = xs.astype(np.float64) * scale
        ys = ys.astype(np.float64) * scale

        s = (xs + ys) * self.F2
        i = np.floor(xs + s).astype(np.int32)
        j = np.floor(ys + s).astype(np.int32)
        t = (i + j).astype(np.float64) * self.G2
        x0 = xs - (i - t)
        y0 = ys - (j - t)

        i1 = np.where(x0 > y0, 1, 0)
        j1 = np.where(x0 > y0, 0, 1)

        x1 = x0 - i1 + self.G2
        y1 = y0 - j1 + self.G2
        x2 = x0 - 1.0 + 2.0 * self.G2
        y2 = y0 - 1.0 + 2.0 * self.G2

        ii = i & 255
        jj = j & 255

        gi0 = self.perm_mod8[ii + self.perm[jj]]
        gi1 = self.perm_mod8[ii + i1 + self.perm[jj + j1]]
        gi2 = self.perm_mod8[ii + 1 + self.perm[jj + 1]]

        t0 = np.maximum(0.5 - x0 * x0 - y0 * y0, 0.0)
        n0 = t0 ** 4 * (self.GRAD2[gi0, 0] * x0 + self.GRAD2[gi0, 1] * y0)

        t1 = np.maximum(0.5 - x1 * x1 - y1 * y1, 0.0)
        n1 = t1 ** 4 * (self.GRAD2[gi1, 0] * x1 + self.GRAD2[gi1, 1] * y1)

        t2 = np.maximum(0.5 - x2 * x2 - y2 * y2, 0.0)
        n2 = t2 ** 4 * (self.GRAD2[gi2, 0] * x2 + self.GRAD2[gi2, 1] * y2)

        return 70.0 * (n0 + n1 + n2)


# ════════════════════════════════════════════════
# 2. fBm 纹理生成
# ════════════════════════════════════════════════

def generate_texture(width=512, height=512, seed=42):
    """
    多八度 Simplex Noise 叠加 + 高频微观粗糙度。
    模拟鼠标垫表面的多尺度纹理。
    """
    simplex = SimplexNoise2D(seed)
    texture = np.zeros((height, width), dtype=np.float64)
    amp = 1.0
    freq = 1.0 / 16.0
    total_amp = 0.0

    for _ in range(6):
        texture += amp * simplex.generate(width, height, scale=freq)
        total_amp += amp
        amp *= 0.5
        freq *= 2.0

    rng = np.random.RandomState(seed + 1)
    texture += rng.randn(height, width) * 8.0

    texture = (texture - texture.min()) / (texture.max() - texture.min()) * 255.0
    return texture.astype(np.uint8)


# ════════════════════════════════════════════════
# 3. 传感器帧截取
# ════════════════════════════════════════════════

def capture_frame(texture, cx, cy, size=18):
    half = size // 2
    h, w = texture.shape
    x0 = int(np.clip(cx - half, 0, w - size))
    y0 = int(np.clip(cy - half, 0, h - size))
    return texture[y0:y0 + size, x0:x0 + size].astype(np.float64)


# ════════════════════════════════════════════════
# 4. FFT 相位相关计算位移
# ════════════════════════════════════════════════

def estimate_displacement(prev, curr):
    """
    频域相位相关：两帧之间的像素位移 (dx, dy)。
    实际鼠标传感器芯片内部使用类似算法。
    """
    cross = np.fft.fft2(prev) * np.conj(np.fft.fft2(curr))
    mag = np.abs(cross)
    mag[mag < 1e-10] = 1e-10
    corr = np.real(np.fft.ifft2(cross / mag))
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    h, w = prev.shape
    dy = peak[0] if peak[0] <= h // 2 else peak[0] - h
    dx = peak[1] if peak[1] <= w // 2 else peak[1] - w
    return dx, dy


# ════════════════════════════════════════════════
# 5. 鼠标传感器模拟器
# ════════════════════════════════════════════════

class MouseSensor:
    """
    模拟光学鼠标传感器。
    内部维护当前位置，每帧拍摄纹理、计算位移、更新位置。
    """

    def __init__(self, texture, start_x=0, start_y=0, sensor_size=18):
        self.texture = texture
        self.sensor_size = sensor_size
        self.x = float(start_x)
        self.y = float(start_y)
        self.prev_frame = capture_frame(texture, start_x, start_y, sensor_size)
        self.trajectory = [(self.x, self.y)]
        self.raw_displacements = []

    def move_to(self, real_x, real_y):
        """
        鼠标真实移动到 (real_x, real_y)。
        传感器拍摄新帧，与上一帧对比计算位移，更新自身估计位置。
        """
        curr_frame = capture_frame(self.texture, real_x, real_y, self.sensor_size)
        dx, dy = estimate_displacement(self.prev_frame, curr_frame)
        self.x += dx
        self.y += dy
        self.trajectory.append((self.x, self.y))
        self.raw_displacements.append((dx, dy))
        self.prev_frame = curr_frame

    def get_metrics(self, real_trajectory):
        """计算传感器估计轨迹与真实轨迹之间的各项误差"""
        n = len(real_trajectory)
        actual_dist = sum(
            np.hypot(real_trajectory[i][0] - real_trajectory[i - 1][0],
                     real_trajectory[i][1] - real_trajectory[i - 1][1])
            for i in range(1, n)
        )
        estimated_dist = sum(np.hypot(dx, dy) for dx, dy in self.raw_displacements)

        point_errors = [
            np.hypot(self.trajectory[i][0] - real_trajectory[i][0],
                     self.trajectory[i][1] - real_trajectory[i][1])
            for i in range(n)
        ]

        total_frames = len(self.raw_displacements)
        exact = sum(
            1 for i in range(total_frames)
            if self.raw_displacements[i][0] == real_trajectory[i + 1][0] - real_trajectory[i][0]
            and self.raw_displacements[i][1] == real_trajectory[i + 1][1] - real_trajectory[i][1]
        )

        return {
            "actual_dist": actual_dist,
            "estimated_dist": estimated_dist,
            "dist_error_pct": abs(estimated_dist - actual_dist) / max(actual_dist, 0.01) * 100,
            "end_offset": np.hypot(self.trajectory[-1][0] - real_trajectory[-1][0],
                                   self.trajectory[-1][1] - real_trajectory[-1][1]),
            "avg_point_error": np.mean(point_errors),
            "max_point_error": np.max(point_errors),
            "exact_frame_pct": exact / total_frames * 100 if total_frames else 0,
            "point_errors": point_errors,
        }


# ════════════════════════════════════════════════
# 6. 测试用例
# ════════════════════════════════════════════════

def make_test_cases():
    cases = []

    # 1. 匀速直线（向右，步长 3px）
    cases.append(("匀速直线", [(5 + i * 3, 200) for i in range(30)]))

    # 2. 圆形轨迹（200 帧，步长 ~2.5px）
    t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    cases.append(("圆形轨迹", [(int(256 + 80 * np.cos(ti)),
                                 int(256 + 80 * np.sin(ti))) for ti in t]))

    # 3. 自由移动（随机方向，步长 2~6px）
    rng = np.random.RandomState(123)
    pts = [(100, 100)]
    for _ in range(49):
        x, y = pts[-1]
        a = rng.uniform(0, 2 * np.pi)
        s = rng.uniform(2, 6)
        pts.append((int(np.clip(x + s * np.cos(a), 20, 490)),
                     int(np.clip(y + s * np.sin(a), 20, 490))))
    cases.append(("自由移动", pts))

    # 4. 大步长移动（步长 8px，测试检测极限）
    cases.append(("大步长移动", [(100 + i * 8, 200) for i in range(30)]))

    return cases


# ════════════════════════════════════════════════
# 7. 主程序
# ════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  光学鼠标传感器模拟")
    print("=" * 60)

    print("\n[1] 生成 Simplex Noise 表面纹理...")
    texture = generate_texture()
    print(f"    尺寸: {texture.shape}")

    cases = make_test_cases()
    print(f"\n[2] {len(cases)} 组测试用例\n")

    fp = "SimHei"
    fig = plt.figure(figsize=(22, 5 * len(cases)))
    fig.suptitle("光学鼠标传感器模拟", fontsize=18, fontweight="bold",
                 fontproperties=fp)
    gs = GridSpec(len(cases), 4, figure=fig, hspace=0.45, wspace=0.3)

    for idx, (name, real_traj) in enumerate(cases):
        sensor = MouseSensor(texture, real_traj[0][0], real_traj[0][1])
        for i in range(1, len(real_traj)):
            sensor.move_to(real_traj[i][0], real_traj[i][1])

        m = sensor.get_metrics(real_traj)

        print(f"  {name:12s} | 帧 {len(real_traj):4d} | "
              f"距离误差 {m['dist_error_pct']:5.1f}% | "
              f"终点偏移 {m['end_offset']:5.1f}px | "
              f"逐点偏差 {m['avg_point_error']:5.1f}px | "
              f"帧精确率 {m['exact_frame_pct']:5.1f}%")

        est = sensor.trajectory

        # ── 列0：表面纹理 + 真实轨迹 ──
        ax0 = fig.add_subplot(gs[idx, 0])
        ax0.imshow(texture, cmap="gray", alpha=0.5)
        ax0.plot([p[0] for p in real_traj], [p[1] for p in real_traj],
                 "g-", linewidth=1.5, label="真实轨迹")
        ax0.plot(real_traj[0][0], real_traj[0][1], "go", markersize=8)
        ax0.set_title(f"纹理 + 真实轨迹\n{name}", fontproperties=fp)
        ax0.legend(prop={"family": fp}, fontsize=7)

        # ── 列1：轨迹对比 ──
        ax1 = fig.add_subplot(gs[idx, 1])
        ax1.plot([p[0] for p in real_traj], [p[1] for p in real_traj],
                 "g-o", markersize=3, linewidth=2, label="真实", alpha=0.7)
        ax1.plot([p[0] for p in est], [p[1] for p in est],
                 "r--s", markersize=3, linewidth=2, label="传感器估计", alpha=0.7)
        ax1.plot(real_traj[0][0], real_traj[0][1], "go", markersize=8)
        ax1.plot(real_traj[-1][0], real_traj[-1][1], "g*", markersize=10)
        ax1.plot(est[-1][0], est[-1][1], "r*", markersize=10)
        ax1.set_title(
            f"轨迹对比\n距离误差 {m['dist_error_pct']:.1f}%  "
            f"终点偏移 {m['end_offset']:.1f}px",
            fontproperties=fp
        )
        ax1.legend(prop={"family": fp}, fontsize=7)
        ax1.invert_yaxis()
        ax1.set_aspect("equal")
        ax1.grid(True, alpha=0.3)

        # ── 列2：逐帧位移对比 ──
        ax2 = fig.add_subplot(gs[idx, 2])
        real_dx = [real_traj[i + 1][0] - real_traj[i][0] for i in range(len(real_traj) - 1)]
        real_dy = [real_traj[i + 1][1] - real_traj[i][1] for i in range(len(real_traj) - 1)]
        est_dx = [d[0] for d in sensor.raw_displacements]
        est_dy = [d[1] for d in sensor.raw_displacements]
        frames = np.arange(len(real_dx))
        w = max(0.3, 3.0 - len(real_dx) * 0.01)
        ax2.bar(frames - w / 2, real_dx, width=w, color="green", alpha=0.6, label="真实 dx")
        ax2.bar(frames + w / 2, est_dx, width=w, color="red", alpha=0.6, label="估计 dx")
        ax2.set_title(f"逐帧 X 位移对比", fontproperties=fp)
        ax2.set_xlabel("帧号", fontproperties=fp, fontsize=8)
        ax2.set_ylabel("dx (px)", fontproperties=fp, fontsize=8)
        ax2.legend(prop={"family": fp}, fontsize=7)
        ax2.grid(True, alpha=0.3)

        # ── 列3：累积偏差曲线 ──
        ax3 = fig.add_subplot(gs[idx, 3])
        ax3.plot(m["point_errors"], "r-", linewidth=1.5)
        ax3.fill_between(range(len(m["point_errors"])), m["point_errors"], alpha=0.2, color="red")
        ax3.axhline(y=m["avg_point_error"], color="orange", linestyle="--",
                     label=f"均值 {m['avg_point_error']:.1f}px")
        ax3.set_title(
            f"累积位置偏差\n最大 {m['max_point_error']:.1f}px  "
            f"帧精确率 {m['exact_frame_pct']:.0f}%",
            fontproperties=fp
        )
        ax3.set_xlabel("帧号", fontproperties=fp, fontsize=8)
        ax3.set_ylabel("偏差 (px)", fontproperties=fp, fontsize=8)
        ax3.legend(prop={"family": fp}, fontsize=7)
        ax3.grid(True, alpha=0.3)

    plt.savefig("mouse_sensor_demo.png", dpi=150, bbox_inches="tight")
    print(f"\n{'=' * 60}")
    print("  结果已保存到 mouse_sensor_demo.png")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
