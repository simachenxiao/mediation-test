# 离线视频分析实验室（独立应用）

位置：`D:/产品/微表情识别/test`

上传一段视频，分别运行**微表情分析**和**生命体征检测**，并逐步查看每一步的中间结果。
另有一个**实时 A / A+ / B 心率对比页**：视频按 1× 播放，A（POS+FFT）、
A+（POS + Welch + SQI，额头优先）和 B（FacePhys.rlap）边播边出结果，
不再等整段视频跑完。

还有一个**确定性重放页**：固定 15fps 抽帧、固定人脸框和 30 秒窗口，
把视频时间传给 A+/B 稳定器，用来验证三路算法本身是否可重复。

以及一个**摄像头实时页 `/camera`**：直接用本地摄像头取景（不读文件），
A / A+ / B 三路同时跑，同时给滚动窗口与全程历史曲线、会话统计（均值/中位数/标准差/相互偏差）。

> 本目录是独立应用，**不会修改 `../code` 里的任何文件**；
> 只以只读方式复用 `../code` 的模型、MediaPipe / onnxruntime-web 运行时和 Python 侧车。

---

## 快速启动

```bash
cd D:/产品/微表情识别/test
npm start
```

浏览器打开：`http://localhost:3200/`

无需在 `test` 目录执行 `npm install`（Server 只用 Node 内置模块）。

前置条件：

1. `../code` 目录存在，且包含：
   - `node_modules/@mediapipe/tasks-vision`
   - `node_modules/onnxruntime-web/dist`
   - `models/`（BlazeFace + 情绪 ONNX）
   - `test/models/blaze_face_full_range.tflite`（本仓库自带，用于更小的人脸）
   - `physio_service/`（rPPG Python 侧车）
2. 生命体征需要 Python 依赖：

```bash
pip install -r ../code/physio_service/requirements.txt
```

微表情不依赖 Python；Python 没装时生命体征会提示侧车未启动，其余功能正常。

### 人脸检测器

`public/lib/face-detect.js` 使用 **MediaPipe BlazeFace Full Range**：

```text
模型：test/models/blaze_face_full_range.tflite
URL ：/local-models/blaze_face_full_range.tflite
```

上一个版本用的是 `../code/models/blaze_face_short_range.tflite`（128×128 输入）。
实测在 1280×720 画面里，人脸低于画面高度约 20% 就会漏检；
换成 full-range 后，约 7%（50px 高）仍能稳定检出，实测 40px 左右才失败。

模型文件随本仓库提交，不修改 `../code`。如果换了模型或想对比，改
`public/lib/face-detect.js` 里的 `MODEL_PATH` 即可。

### 实时 A / A+ / B 对比（B 引擎）

打开 `http://localhost:3200/live`，需要额外准备：

- `../EmotiEffLib-main/rppg-demo/rppg-demo/vendor/open_rppg`（FacePhys.rlap 权重与代码）
- Python 侧已安装 `jax`、`keras`、`onnxruntime`、`opencv-python`、`heartpy`

```bash
pip install "jax" "keras>=3" onnxruntime opencv-python heartpy
```

B 模型首次加载约 **20–40 秒**，页面会显示加载状态；加载完成后才能开始实时分析。
A/A+/B 都使用 30 秒滑动窗口，视频播放期间会禁用进度条拖动。

A+ 是纯 NumPy/SciPy 侧车，不需要额外模型或新依赖；如果生命体征依赖已安装，A+ 就能启动。A+ 从约 12 秒开始，之后每约 1 秒更新一次；A 从约 8 秒开始，之后也每约 1 秒更新。

A+ 现在只使用 POS：额头优先；只有额头没有可用候选时，才回退到脸颊。

### 确定性重放

打开 `http://localhost:3200/replay`。

它不按播放时钟采样，而是：

- 固定 15fps seek 抽帧；
- 每 5 帧检测一次人脸并复用固定框；
- 在固定检查点构造同一份 30 秒 payload；
- A/A+ 自动复跑两次并逐点比较；
- A+/B 的稳定器使用 `replayTime`（视频时间），不用 wall clock；
- B 需要点“再跑一次并对比”验证第二次。

这不是更准的测量，而是把“输入变化”和“算法变化”分开。

### 摄像头实时对比

打开 `http://localhost:3200/camera`。

- 用 `getUserMedia` 读本地摄像头，页面自动镜像（自拍视角）；
- 会话时钟从点「开启摄像头」开始，与视频 currentTime 无关；
- A/A+ 每秒出一次结果，B 每约 1 秒轮询 `/deep/result`；
- 四张图：滚动 60s 心率、全程心率、滚动 60s 质量、原始候选 HR；
- 会话统计：均值 / 中位数 / 最小最大值 / 标准差，以及 A+/A、B/A、B/A+ 的平均绝对差；
- 可叠加人脸框与额头/脸颊 ROI，并给出运动门控提示；
- 「导出 JSON」保存本次会话的逐点序列与统计。

浏览器要求：页面必须运行在 `http://localhost` 或 `https`（`localhost` 属于安全上下文），
否则 `navigator.mediaDevices` 不可用。摄像头画面只在浏览器本地处理，不上传服务器；
发给本机 Python 侧车的只有 ROI 平均颜色（A/A+）和 JPEG 帧（B）。

注意：摄像头输入没有固定真值，头动、光照变化、自动曝光都会影响结果；
A/A+ 依赖人脸框取 ROI，B 走整帧深度模型，所以三路不一致是正常现象。

### 可配置环境变量

```bash
PORT=3201 npm start                 # 换端口（默认 3200）
HOST=127.0.0.1 npm start            # 只允许本机访问（默认 0.0.0.0）
PHYSIO_PYTHON=python3.11 npm start  # 指定 Python 解释器
APLUS_PYTHON=python3.11 npm start    # 指定 A+ 侧车解释器（可选）
```

---

## 文件结构

```
test/
├── server.mjs                  独立 HTTP 服务（Node 内置模块，无 npm 依赖）
├── deep_service.py             B 引擎侧车：Open-rppg FacePhys.rlap
├── aplus_service.py            A+ 侧车：POS + Welch + SQI
├── package.json                npm start -> node server.mjs
├── README.md                   本文件
├── models/
│   └── blaze_face_full_range.tflite  full-range 人脸检测模型
├── docs/
│   ├── 离线视频分析实验室.md    详细说明 / 模块 API
│   └── 方法A的问题与A+改进.md   A 的问题、A+ 设计、确定性重放
└── public/
    ├── lab.html                离线分步分析页
    ├── lab.js                  离线交互、步骤状态、图表、导出
    ├── live.html               实时 A / A+ / B 对比页
    ├── live.js                 实时播放、A/A+/B 采集与图表
    ├── replay.html             确定性重放页
    ├── replay.js               固定帧、固定窗口、复跑验证
    ├── camera.html             摄像头实时 A / A+ / B 对比页
    ├── camera.js               摄像头取景、三路采集、历史曲线、统计
    ├── stat-live.js            会话统计（均值/中位数/偏差）
    └── lib/
        ├── emotion.js          微表情模块（独立）
        ├── vitals.js           生命体征模块（独立）
        ├── face-detect.js      MediaPipe BlazeFace Full Range 检测
        ├── roi.js              面部 ROI 像素采样
        ├── physio-live.js      实时 A 采样（30s 窗口，A+ 复用）
        ├── video-frames.js     抽帧 / seek / 重采样
        └── charts.js           canvas 图表
```

运行时路径映射（只读）：

| URL | 实际目录 |
|-----|----------|
| `/mediapipe/*` | `../code/node_modules/@mediapipe/tasks-vision/*` |
| `/wasm/*` | `../code/node_modules/@mediapipe/tasks-vision/wasm/*` |
| `/ort/*` | `../code/node_modules/onnxruntime-web/dist/*` |
| `/models/*` | `../code/models/*` |
| `/local-models/*` | `test/models/*` |
| `/physio/estimate` | 代理到本机 Python 侧车 `../code/physio_service` |
| `/live` | 实时 A / A+ / B 对比页 |
| `/replay` | 确定性重放页 |
| `/camera` | 摄像头实时 A / A+ / B 对比页 |
| `/aplus/estimate` | 代理到 `aplus_service.py`（A+ 纯信号处理侧车） |
| `/deep/*` | 代理到 `deep_service.py`（FacePhys.rlap） |

---

## 分步结果

### 微表情（左侧栏）

1. 视频解码与抽帧规划：时长、分辨率、目标帧率、计划帧数、首帧预览。
2. 人脸检测：逐采样帧人脸框，可拖动滑块回看，显示检出率。
3. 人脸裁剪与归一化：逐帧缩略图、224×224 输入、RGB 均值。
4. 情绪推理：8 类情绪概率、引擎/耗时、逐帧结果表。
5. 微表情时间线：主导情绪时间带、概率曲线、负向占比、短时表情事件。

### 生命体征（右侧栏）

1. 人脸跟踪与 ROI 定位：人脸框 + 额头/左颊/右颊 ROI 叠加。
2. ROI 像素采样曲线：R/G/B 或归一化绿通道。
3. 分窗口信号估计：逐窗口调用 `/physio/estimate`，追加 HR/RR/SpO₂/质量。
4. 生理指标汇总：HR/RR/SpO₂/血压/HRV/信号质量，及随窗口变化曲线。

> SpO₂ 与血压为普通 RGB 摄像头下的**估计值**，非医疗级，系数需逐设备、逐人标定。

---

## 侧车状态

页面右侧标题旁会显示：

- `生理侧车运行中 :端口` → 正常
- `生理侧车未运行` → 点「检查」会自动尝试重启

也可手动检查：

```bash
curl http://localhost:3200/physio/health
```

如果报缺少 `numpy` / `scipy`，执行：

```bash
pip install -r ../code/physio_service/requirements.txt
```

---

## 说明

- 视频只在浏览器本地解码，不上传；只有 ROI 颜色数值会发给本机 Python 侧车。
- 建议视频：10 秒以上、正脸清晰、光照均匀、人脸尽量少晃动（rPPG 对运动敏感）。
- 两个模块都支持 JSON / CSV 导出。
- 实时页每次按播放采样，结果天然会有波动；要验证同一视频的算法可重复性，用 `/replay`。
- 摄像头页需要 `localhost` 或 `https` 才能授权；无摄像头时可用 Chrome 的
  `--use-fake-device-for-media-stream` 做界面自测。
- 更详细的算法拆分与模块 API 见 [docs/离线视频分析实验室.md](docs/离线视频分析实验室.md)。
- A 的六个问题与 A+ 设计见 [docs/方法A的问题与A+改进.md](docs/方法A的问题与A+改进.md)。
