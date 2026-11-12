# kneo-roi-line-alert（Xtion 版）

> 這是使用 **ASUS Xtion PRO Live（OpenNI2）** 的版本。USB webcam 版在隔壁 `kneo-roi-line-alert` 資料夾；兩版共用的程式用上一層的 `sync_versions.py` 從 webcam 版同步過來。

KNEO Pi 接 ASUS Xtion PRO Live（透過 OpenNI2 讀彩色影像），偵測到「有人走進 ROI（指定區域）」時，透過 **LINE Messaging API** 傳通知（可附即時截圖）。
人物偵測（YOLOv8n）與人臉偵測（UltraFace）編譯成 **NEF，在 KL730 NPU 上執行**。

> ⚠️ LINE Notify 已於 2025/3/31 停止服務，本專案用的是 Messaging API（官方帳號 bot 推播）。

```
Xtion ──▶ KL730 NPU（NEF）: YOLOv8n 人物 ／ UltraFace 人臉 ─▶ CPU：解碼 + NMS
       ─▶ ROI 判斷 ─▶ 人數統計（同時進／陸續進）+ 冷卻 ─▶ LINE push（文字 + 截圖）
       └─▶ 網頁即時畫面、網頁修改 ROI（--stream）
```

## 專案結構

| 路徑 | 說明 |
|---|---|
| `kneo_alert/app.py` | 主程式：讀影格 → 偵測（背景執行緒）→ ROI → 觸發 → 通知；攝影機斷線自動重連 |
| `kneo_alert/camera.py` | 攝影機設定與開啟（`openni2` = Xtion、USB webcam、影片檔、RTSP） |
| `kneo_alert/_openni_compat.py` | 讓 `openni` 套件能在 Python 3.14 上 import 的修正 |
| `kneo_alert/detector.py` | 偵測器工廠：`kneron_nef`、`cut_onnx`、`yolo_onnx`、`hog` |
| `kneo_alert/npu_detector.py` | KL730 NPU 推論（Kneron PLUS `kp`）、人臉確認 |
| `kneo_alert/yolo_post.py`、`face_post.py` | NPU 輸出的 CPU 後處理（sigmoid / DFL softmax / 框解碼） |
| `kneo_alert/roi.py` | 多邊形 ROI，`foot`（腳點）或 `overlap`（重疊比例）判斷 |
| `kneo_alert/trigger.py` | 人數統計狀態機：同時進、陸續進都算，算齊後傳一則；冷卻時間 |
| `kneo_alert/notifier.py` | LINE push（重試、防重複）、imgbb 上傳截圖、背景執行緒送出 |
| `kneo_alert/stream.py` | 網頁即時畫面與 ROI 編輯器（`--stream`） |
| `npu/` | 轉 NEF 的工具：切模型、拍校正圖片、量化並編譯 |
| `tools/` | `pick_roi.py`（電腦上點 ROI）、`send_test_message.py`、`line_webhook_id.py` |
| `deploy/` | `setup_pi.sh`（板子安裝與檢查）、`build_openni2.sh`（aarch64 編譯 OpenNI2）、`check_openni2.py`、systemd |

## 1. 安裝

板子（KNEO Pi，Arch Linux ARM）：

```bash
cd ~/kneo-roi-line-alert-xtion
bash deploy/setup_pi.sh
```

腳本會建立 venv（有系統內建 OpenCV 時直接沿用，並可使用板子預裝的 Kneron PLUS `kp`）、安裝 `requirements.txt`、
建立 `config.yaml` / `.env`、測試模型與攝影機。**不要在板子上執行 `pacman -Syu`**（廠商映像檔的套件庫是舊的）。

### Xtion（OpenNI2）

Linux 沒有 Xtion 的 V4L2 驅動（不會出現 `/dev/video*`），要透過 OpenNI2 讀取：

```bash
bash deploy/build_openni2.sh                     # 編譯 OpenNI2 + PS1080 驅動（含 aarch64 修正），裝到 /opt/OpenNI2
sed -i 's/^  source: .*/  source: openni2/' config.yaml
.venv/bin/python deploy/check_openni2.py         # 抓 30 張彩色影像測 fps，存 camera_test.jpg
```

`build_openni2.sh` 只會安裝缺少的編譯工具，不會升級系統套件。

電腦（開發、測試用）：

```bash
python -m venv .venv
.venv\Scripts\activate             # Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
copy config.example.yaml config.yaml
copy .env.example .env
```

## 2. LINE 設定

1. 到 [LINE Developers Console](https://developers.line.biz/console/) 建立 Provider（名稱用英文即可）。
   自 2024/9 起不能在 Console 直接建立 Messaging API channel，要：
   1. 點 **Create a Messaging API channel** → 跳到 LINE Official Account Manager 建立官方帳號。
   2. 在 Official Account Manager 的 **設定 → Messaging API** 按「啟用 Messaging API」，選剛建立的 Provider。
   3. 回到 Developers Console，Provider 底下就會出現這個 channel。
2. **Messaging API** 分頁 → 最下方 **Channel access token (long-lived)** → Issue，貼到 `.env` 的 `LINE_CHANNEL_ACCESS_TOKEN`。
3. 取得接收者 ID 填進 `LINE_TO`：
   - **傳給自己**：**Basic settings** 分頁最下方的 **Your user ID**（`U` 開頭）。用手機掃 Messaging API 分頁的 QR code 把 bot 加為好友。
   - **傳到群組**：群組 ID（`C` 開頭）只能從 webhook 事件取得：
     1. 允許 bot 加入群組（LINE Official Account Manager → 設定 → 帳號設定）。
     2. `python tools/line_webhook_id.py --port 8000`
     3. 另開終端機用 `ngrok http 8000`（或 `cloudflared tunnel --url http://localhost:8000`）取得 HTTPS 網址。
     4. Webhook URL 填 `https://<網址>/callback`，打開 **Use webhook**。
     5. 把 bot 邀進群組或在群組裡說話，終端機會印出 `LINE_TO=C...`。
4. 測試：`python tools/send_test_message.py`

**免費方案每月推播則數有上限**（傳到群組時每個成員都算一則）。測試時一律加 `--dry-run`，並用 `trigger.cooldown_sec` 控制頻率。

## 3. 執行

```bash
python -m kneo_alert --dry-run --stream   # 不傳 LINE，瀏覽器開 http://<板子IP>:8080/ 看即時畫面
python -m kneo_alert --stream             # 正式執行
python -m kneo_alert --source video.mp4 --dry-run   # 用錄好的影片測試
```

畫面中：黃框是 ROI，紅框是 ROI 內的人，綠框是 ROI 外的人，藍框是人臉（開啟 `require_face` 時），左上角是人數、狀態與 fps。

> 網頁沒有密碼保護，只在信任的區域網路（或 Tailscale 這類私人網路）使用，不要把這個埠開到外網。

### 修改偵測範圍（ROI）

用 `--stream` 執行時，瀏覽器開 `http://<板子IP>:8080/roi`：按「清除」→ 依序點出範圍（至少 3 點，手機也能點）→「儲存」。
**立即生效**，並寫回板子上的 `config.yaml`。（有螢幕的電腦也可以用 `tools/pick_roi.py`。）

### 部署到板子（從電腦）

```powershell
tar --exclude=.venv --exclude=snapshots --exclude=calib --exclude=__pycache__ --exclude=.pytest_cache -czf ..\kneo.tgz .
scp ..\kneo.tgz <user>@<ip>:~/
ssh <user>@<ip> "mkdir -p ~/kneo-roi-line-alert-xtion && tar -xzf ~/kneo.tgz -C ~/kneo-roi-line-alert-xtion"
```

（`.env`、`config.yaml`、`models/` 會一起帶過去。Windows 內建 `scp` 遇到中文路徑可能失敗，請先 `cd` 到專案資料夾、目的地用相對路徑。）

開機自動啟動：`bash deploy/setup_pi.sh --install-service`，看 log：`journalctl -u kneo-roi-alert -f`。

## 4. 人數統計：同時進、陸續進都算

第一個人進入 ROI 後，系統會繼續算 `trigger.gather_sec` 秒（預設 5），這段時間內**每一波進來的人都會加總**，
時間到才傳**一則**「N 人」。

| 情況 | 通知 | 截圖（`attach_image: true`） |
|---|---|---|
| 兩人一起進來 | 2 人 | 1 張（兩人都在） |
| 窄門：一人進、離開框，另一人再進 | 2 人 | 2 張 |
| 兩人一起進，第三人之後才到 | 3 人 | 2 張 |
| 一人在框內時另一人加入 | 2 人 | 1 張（換成兩人都在的那張） |

- 文字 + 最多 4 張截圖在同一次推播送出，只算 1 則額度。
- 人數要**持續變少 `leave_sec` 秒（預設 1）**才算有人離開，避免框在 ROI 邊緣閃爍被算成好幾個人。
  代價是：前一人離開後 1 秒內後一人就補上，會被算成同一人。
- 系統只數人頭、不認人：同一人離開超過 1 秒再進來，在統計時間內會被算兩次。
- 兩人前後重疊（擋住）時偵測器可能只框到 1 人。

## 5. 通知附上即時截圖（imgbb）

LINE 傳圖片必須是公開的 HTTPS 網址，所以截圖會先上傳到 [imgbb](https://imgbb.com/) 再傳到 LINE。

1. 到 https://imgbb.com/ 註冊並登入，再到 https://api.imgbb.com/ 取得 API key。
2. 在板子的 `.env` 加上 `IMGBB_API_KEY=你的key`。
3. `config.yaml` 設定 `alert.attach_image: true`；`image_expiration_sec` 是 imgbb 自動刪除時間（預設 7 天）。

- 有人的照片會上傳到第三方網站（網址很難猜，但知道網址的人都看得到）。
- 上傳失敗時仍會傳文字通知。`--dry-run` 不會上傳。

## 6. NPU（KL730）與 NEF

板子上的 KL730 透過 USB 連接，用預裝的 Kneron PLUS（`kp`）載入 NEF。
`models/models_730.nef` 同時包含兩個模型（model id **32769** = 人物、**32770** = 人臉）。

| | CPU（YOLOv8n 640） | **NPU（YOLOv8n 320，NEF）** |
|---|---|---|
| 每張耗時（板子實測） | 約 994 ms | **約 55 ms**（含前處理、USB 傳輸、後處理） |

### 設定

```yaml
detector:
  type: kneron_nef              # 板子：NPU；電腦測試用 cut_onnx（CPU 跑同樣切開的模型）
  input_size: 320
  nef_path: models/models_730.nef
  firmware_path: models/kp_firmware.tar
  require_face: false           # true = 人物框上半部要偵測到臉才算數（見下方）
```

`kp_firmware.tar` 從 Kneron 官方範例複製（每次連線會上傳到 KL730）：

```bash
git clone --depth 1 https://github.com/kneron/kneopi-examples.git ~/kneopi-examples
cp ~/kneopi-examples/ai_application/plus_python/res/firmware/KL730/kp_firmware.tar models/
```

啟動時出現 `[warnning] The version of firmware ... is not the corresponding version` 可忽略，推論正常。

### 人臉確認（`require_face: true`）

人物框的**上半部**要偵測到臉才算一個人。可以擋掉沒有臉的人形誤判（掛著的衣服、椅子、立牌背面）。
**擋不掉人臉照片**，而且**背對鏡頭的人不會被計入**，鏡頭要能拍到走進來的人的正面。

### 自己轉 NEF

需要 Docker 與 `kneron/toolchain` 映像檔（v0.33.1 測試過）。在電腦上：

```bash
# 1. 匯出並切掉後處理（NPU 只跑卷積；sigmoid / softmax / 解碼改在 CPU）
pip install ultralytics onnx
yolo export model=yolov8n.pt format=onnx imgsz=320 opset=12        # -> yolov8n.onnx（改名 models/yolov8n_320.onnx）
python npu/cut_yolov8.py models/yolov8n_320.onnx npu/yolov8n_320_cut.onnx
#    人臉：下載 UltraFace version-RFB-320.onnx（MIT，Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB）
python npu/cut_ultraface.py version-RFB-320.onnx npu/ultraface_320_cut.onnx
```

```bash
# 2. 在板子上拍 100 張實際場景（有人、沒人、遠近、正側背面都要有），再抓回電腦
.venv/bin/python npu/capture_calib.py --count 100 --interval 1.0
```

```bash
# 3. 量化 + 兩個模型編進同一個 NEF。<work> 放：兩個 *_cut.onnx、kneo_alert/yolo_post.py、
#    kneo_alert/face_post.py、npu/compile_nef.py、calib/*.jpg（路徑請用英文）
docker run --rm -v <work>:/data kneron/toolchain:latest \
  /workspace/miniconda/envs/onnx1.13/bin/python /data/compile_nef.py --work /data
# -> <work>/nef/models_730.nef，複製到 models/
```

腳本會印出兩個模型的 NPU 評估（是否有層退回 CPU）以及量化前後的分數比較。
換地點（例如從家裡搬回實驗室）時，建議用新場景的畫面重拍校正圖片、重新編譯。

## 7. 參數調整

| 現象 | 調整 |
|---|---|
| 沒人也通知（誤判） | 提高 `conf_threshold`、`enter_frames`；開啟 `require_face` |
| 有人卻沒通知 | 降低 `conf_threshold`；ROI 是否涵蓋人的位置（`foot` 看腳、`overlap` 看重疊）；`require_face` 時要拍得到臉 |
| 一個人被算成好幾個 | 提高 `leave_sec`；ROI 不要畫在人常站的邊界上 |
| 緊跟著進來的兩人被算成一個 | 降低 `leave_sec`（例如 0.5） |
| 同一批人通知太多次 | 提高 `exit_sec`、`cooldown_sec` |
| 陸續進來的人沒算到 | 提高 `gather_sec`（例如 8～10） |

## 測試

```bash
pip install -r requirements-dev.txt
pytest
```

## 已知限制

- 系統數人頭、不追蹤個人身分（見第 4 節）。
- 人臉確認擋不掉照片；要分辨照片與真人需要深度資訊（例如 Xtion 版的深度鏡頭）。
- 截圖經由第三方圖床（imgbb）傳給 LINE。
