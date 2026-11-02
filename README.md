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

> 本目录是**完全独立应用**：MediaPipe / onnxruntime-web / 模型 / Physio 侧车
> 都在本目录的 `vendor/` 和 `physio_service/` 里，不再读取 `../code`
> 或 `../EmotiEffLib-main`。

---

## 快速启动

```bash
cd D:/产品/微表情识别/test
npm start
```

浏览器打开：`http://localhost:3200/`

无需在 `test` 目录执行 `npm install`（Server 只用 Node 内置模块）。

Linux 上建议：

```bash
cd test
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash start.sh
```

`start.sh` 会自动使用 `.venv` 里的 Python，并设置 A/A+/B 三个侧车的解释器。
Windows 继续用 `start.bat` 或 `npm start`。

前置条件：

1. 本目录已经内置：
   - `vendor/mediapipe/`（MediaPipe tasks-vision）
   - `vendor/ort/`（onnxruntime-web 的 WebGPU / WASM 文件）
   - `vendor/models/`（BlazeFace short + 情绪 ONNX）
   - `vendor/open_rppg/`（FacePhys.rlap）
   - `vendor/vital_signs/`（B 的 HRV/呼吸估计）
   - `models/blaze_face_full_range.tflite`
   - `physio_service/`（rPPG Python 侧车）
2. 生命体征需要 Python 依赖：

```bash
pip install -r requirements.txt
```

微表情不依赖 Python；Python 没装时生命体征会提示侧车未启动，其余功能正常。

### 人脸检测器

`public/lib/face-detect.js` 使用 **MediaPipe BlazeFace Full Range**：

```text
模型：test/models/blaze_face_full_range.tflite
URL ：/local-models/blaze_face_full_range.tflite
```

上一个版本用的是 `vendor/models/blaze_face_short_range.tflite`（128×128 输入）。
实测在 1280×720 画面里，人脸低于画面高度约 20% 就会漏检；
换成 full-range 后，约 7%（50px 高）仍能稳定检出，实测 40px 左右才失败。

模型文件随本仓库提交，不再依赖 `../code`。

`/camera` 页面的「人脸模型」下拉框可以在运行中切换：

- Full Range（小脸）：默认，覆盖更远；
- Short Range（原版）：`vendor/models/blaze_face_short_range.tflite`。

页面会按模型分别累计检测次数、命中率和中位耗时，切换后可以直接对比。
驱动逻辑在 `public/lib/face-detect.js` 的 `FACE_MODELS` / `getFaceDetector(modelId)`；
其他页面默认使用 Full Range。

### 实时 A / A+ / B 对比（B 引擎）

打开 `http://localhost:3200/live`，需要额外准备：

- `vendor/open_rppg/`（FacePhys.rlap 权重与代码，本目录已内置）
- Python 侧已安装 `jax`、`keras`、`onnxruntime`、`opencv-python`、`heartpy`

```bash
pip install "jax" "keras>=3" onnxruntime opencv-python heartpy
```

B 模型首次加载约 **20–90 秒**（首次 JAX 编译可能更久），页面会显示加载状态；加载完成后才能开始实时分析。
A/A+/B 都使用 30 秒滑动窗口，视频播放期间会禁用进度条拖动。

A+ 是纯 NumPy/SciPy 侧车，不需要额外模型或新依赖；如果生命体征依赖已安装，A+ 就能启动。A+ 从约 12 秒开始，之后每约 1 秒更新一次；A 从约 8 秒开始，之后也每约 1 秒更新。

A+ 现在只使用 POS：额头优先；只有额头没有可用候选时，才回退到脸颊。
A+ 也会返回 `respiratory_rate` / `respiratory_confidence`：从当前主 ROI 的原始
绿色通道估计，算法与 A 的 respiratory 同源（0.1–0.5Hz + 脉搏包络）；
页面 A+ 卡片会显示「呼吸 RR / RR 置信度」。

A 的 `/physio/estimate` 同时返回呼吸率（RR）和 RR 置信度：
`/live` 和 `/camera` 的 A 卡片会显示「呼吸 RR / RR 置信度」，
`/camera` 还会画出 RR 曲线。呼吸率来自 A 的 forehead 原始绿色通道。

B 的呼吸率来自本目录 `vendor/vital_signs` 的 `HrvRespirationEstimator`：先读 HRV 的
`breathingrate`，没有就从 BVP 原始信号做 0.08–0.5Hz 带通 + Welch。
`/live` 和 `/camera` 的 B 卡片会显示「B 呼吸 RR / RR 来源」，来源分为
`HRV 派生`、`BVP 派生`、`保持`。B 的 RR 是实验性输出，不等于医学呼吸监测。
A+ 的 RR 与 A 使用同一类绿色通道模型，但主 ROI 可能是 forehead 或回退后的 cheeks。

B 的人脸检测是 open_rppg **内部**的 BlazeFace ONNX（默认 `blaze_face_full.onnx`，192×192，每 5 帧检测一次），
不是页面上可切换的 MediaPipe 模型。可用 `DEEP_FACE_MODE=Near` 切回 open_rppg 原来的
`blaze_face.onnx`（128×128）。B 卡片里的「B 人脸」显示它自己的命中框大小；
摄像头页会用紫色虚线画出 B 的框，方便和 A/A+ 的浏览器检测框对比。

### B 的心率/呼吸实现与 rppg-demo 的关系

B 与参考实现 rppg-demo 的 HR 核心逻辑一致：

- 都用 `Model.hr(start=-30, return_hrv=False)` 取 `hr` 和 `SQI`；
- 都用同一套 `HeartRateStabilizer`（35–190bpm 范围、SQI<0.2 预热、
  SQI<0.18 保持、5bpm/s 限速、`alpha=0.12+0.18*quality`）；
- B 只是额外把 `now` 传进去，用于 `/replay` 的确定性重放。

不同的地方：

- rppg-demo 还计算 HRV、BP、SpO2 和 BVP 波形，B 只暴露 HR/RR/人脸；
- rppg-demo 的 HR 窗口可通过 `hrWindow` 配，B 固定 30s；
- B 的 `face_mode` 现在是 `Far`（192px），rppg-demo 默认是 `Near`（128px）；
- B 使用本目录 `vendor/vital_signs` 的 `HrvRespirationEstimator` 来产出 RR；
  该实现来自 rppg-demo，但已经内置到本目录。

B 的 RR 用 45s 窗口，优先 HRV 的 `breathingrate`，失败时从 `model.bvp(raw=True)`
做 0.08–0.5Hz 带通 + Welch；因此 B 的 RR 与 A 的额头绿色通道 RR 不是同一条信号。

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
- A 卡片显示心率、置信度、呼吸 RR 和 RR 置信度；A+ 卡片显示 A+ 的 RR 与 RR 置信度；B 卡片显示 B 的呼吸 RR 与来源；
- 五张图：滚动 60s 心率、全程心率、滚动 60s 质量、原始候选 HR、A/B 呼吸率趋势；
- 会话统计：均值 / 中位数 / 最小最大值 / 标准差，以及 A+/A、B/A、B/A+ 的平均绝对差；
- 可叠加人脸框与额头/脸颊 ROI，并给出运动门控提示；
- 可运行中切换人脸模型（Full Range / Short Range），并按模型累计命中率与中位耗时；
- 叠加 B 自己的内部人脸框（紫色虚线），看 B 是否也检测到同一张脸；
- 显示 A 的呼吸率（RR）与 RR 趋势；B 卡片也显示 B 的 RR（实验性）；
- 「导出 JSON」保存本次会话的逐点序列、统计与人脸模型统计。

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
DEEP_FACE_MODE=Near npm start        # B 内部检测器切回 128px（默认 Far=192px）
DEEP_RR_WINDOW=45 npm start          # B 呼吸率窗口（默认 45s）
```

---

## 文件结构

```
test/
├── server.mjs                  独立 HTTP 服务（Node 内置模块，无 npm 依赖）
├── deep_service.py             B 引擎侧车：Open-rppg FacePhys.rlap
├── aplus_service.py            A+ 侧车：POS + Welch + SQI
├── requirements.txt            Linux/Python 全部依赖
├── start.sh                    Linux 启动脚本
├── package.json                npm start -> node server.mjs
├── README.md                   本文件
├── models/
│   └── blaze_face_full_range.tflite  full-range 人脸检测模型
├── vendor/
│   ├── mediapipe/              MediaPipe tasks-vision
│   ├── ort/                    onnxruntime-web 浏览器运行时
│   ├── models/                 情绪 ONNX + BlazeFace short
│   ├── open_rppg/              FacePhys.rlap + B 内部人脸模型
│   └── vital_signs/            B 的 HRV/呼吸估计
├── physio_service/             A 的 rPPG Python 侧车
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
        ├── face-detect.js      MediaPipe BlazeFace（full/short 可切换）
        ├── roi.js              面部 ROI 像素采样
        ├── physio-live.js      实时 A 采样（30s 窗口，A+ 复用）
        ├── video-frames.js     抽帧 / seek / 重采样
        └── charts.js           canvas 图表
```

运行时路径映射（全部在本目录内）：

| URL | 实际目录 |
|-----|----------|
| `/mediapipe/*` | `vendor/mediapipe/*` |
| `/wasm/*` | `vendor/mediapipe/wasm/*` |
| `/ort/*` | `vendor/ort/*` |
| `/models/*` | `vendor/models/*` |
| `/local-models/*` | `models/*` |
| `/physio/estimate` | 代理到本机 Python 侧车 `physio_service/` |
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
pip install -r requirements.txt
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
