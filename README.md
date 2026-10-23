# 离线视频分析实验室（独立应用）

位置：`D:/产品/微表情识别/test`

上传一段视频，分别运行**微表情分析**和**生命体征检测**，并逐步查看每一步的中间结果。
两条链路完全独立：可以只跑其中一个，也可以先跑微表情再跑生命体征。

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
   - `physio_service/`（rPPG Python 侧车）
2. 生命体征需要 Python 依赖：

```bash
pip install -r ../code/physio_service/requirements.txt
```

微表情不依赖 Python；Python 没装时生命体征会提示侧车未启动，其余功能正常。

### 可配置环境变量

```bash
PORT=3201 npm start                 # 换端口（默认 3200）
HOST=127.0.0.1 npm start            # 只允许本机访问（默认 0.0.0.0）
PHYSIO_PYTHON=python3.11 npm start  # 指定 Python 解释器
```

---

## 文件结构

```
test/
├── server.mjs                  独立 HTTP 服务（Node 内置模块，无 npm 依赖）
├── package.json                npm start -> node server.mjs
├── README.md                   本文件
├── docs/
│   └── 离线视频分析实验室.md    详细说明 / 模块 API
└── public/
    ├── lab.html                页面
    ├── lab.js                  交互、步骤状态、图表、导出
    └── lib/
        ├── emotion.js          微表情模块（独立）
        ├── vitals.js           生命体征模块（独立）
        ├── face-detect.js      MediaPipe 人脸检测
        ├── roi.js              面部 ROI 像素采样
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
| `/physio/estimate` | 代理到本机 Python 侧车 `../code/physio_service` |

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
- 更详细的算法拆分与模块 API 见 [docs/离线视频分析实验室.md](docs/离线视频分析实验室.md)。
