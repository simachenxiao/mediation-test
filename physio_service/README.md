# physio_service —— lifemirror 的 rPPG 生理指标计算侧车

本目录把 `rppg` 工程中**纯 CPU、纯 Python** 的生理指标算法（`src/physio`）搬过来，
封装成一个本地 HTTP 微服务。Electron/Node 在需要时懒启动它，前端只需调用同源的
`/physio/estimate`。**不涉及任何 NPU / 模型 / OpenCV。**

## 分工

```
MediaPipe 人脸检测(JS) ─► ROI 像素采样(JS，YCbCr 肤色) ─► 固定 fs 重采样(JS)
        │                                                        │
        │                                   POST /physio/estimate │
        ▼                                                        ▼
   情绪 ONNX 推理                                   physio_service（numpy+scipy）
   （完全独立、互不阻塞）                            └─ estimate_all() → 指标 JSON
```

- 人脸检测继续用 lifemirror 已有的 **MediaPipe BlazeFace**，无需新模型。
- Python 侧只接收 `/physio/estimate` 的 payload：`fs` + `timestamps` + 三路 ROI 的
  `(R,G,B)` 均值时序；返回 HR / 心率变化率 / RR / SpO₂ / 血压 / HRV / 质量。
- 服务 **stateless**：每次传整段滑动窗口，服务端无会话，崩溃重启无影响。

## 目录

| 文件 | 说明 |
|------|------|
| `server.py` | HTTP 服务；`GET /health`、`POST /estimate`、`POST /shutdown`；启动时 stdout 打印 `PHYSIO_PORT=<port>` |
| `run.py` | PyInstaller 打包入口 |
| `config.json` | 算法参数与校准系数（来自 rppg `config.yaml` 的 `rppg` 段） |
| `physio/` | 从 rppg `src/physio` 原样搬运的算法 |
| `utils/metrics.py` | 从 rppg `src/utils/metrics.py` 搬运的工具 |
| `test_synthetic.py` | 合成信号自检（无需摄像头/Electron） |
| `physio_service.spec` | PyInstaller 配置（onedir，输出 `dist/physio_service/`） |

## 开发运行

```bash
# 依赖（numpy 必须 <2，与 rppg 保持一致；scipy 提供 butter/welch/find_peaks）
pip install -r requirements.txt

# 自检（合成信号，验证算法搬运无损）
python3 -m physio_service.test_synthetic     # 在 lifemirror 根目录执行

# 手动启动服务（一般不用，由 Node/Electron 自动拉起）
python3 -m physio_service.server --port 8765
```

## 打包（Linux）

```bash
# 在 lifemirror 根目录
npm run build:physio          # 生成 physio_service/dist/physio_service/physio_service
npm run dist:linux            # 自动先 build:physio，再 electron-builder
```

产物 `release/*.AppImage` 内已含自带 Python 运行时，**目标机器无需安装 Python**。

> ⚠️ PyInstaller 不支持交叉编译：Linux 侧车必须在 Linux 上构建；Windows/macOS
> 同理需各自的构建环境。若只在 Linux 使用则无此问题。
>
> 开发机若报错 `Python shared library not found`，安装 `libpython3.11`；
> 若 `venv` 创建失败，安装 `python3.11-venv`。

## 接口

`POST /estimate`

```json
{
  "fs": 30.0,
  "timestamps": [0.0, 0.0333, ...],
  "forehead":    [[r,g,b], ...],
  "left_cheek":  [[r,g,b], ...],
  "right_cheek": [[r,g,b], ...]
}
```

返回字段与 rppg `doc/08-生理指标估计说明.md` 一致（`heart_rate`、`respiratory_rate`、
`spo2`、`blood_pressure`、`hrv`、`heart_rate_change_bpm_per_min`、`quality` 等）。
数组中 `null` 表示该帧 ROI 无效，算法会跳过。

## 重要声明

SpO₂ 与血压为普通 RGB 摄像头下的**估计值**，非医疗级；系数（`config.json` 的
`spo2` / `blood_pressure`）必须逐设备、逐人用参考仪器标定。
