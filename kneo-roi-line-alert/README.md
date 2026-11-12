# kneo-roi-line-alert

攝影機偵測到「有人走進 ROI（指定區域）」時，透過 **LINE Messaging API** 傳訊息通知。

> ⚠️ LINE Notify 已於 2025/3/31 停止服務，本專案用的是 Messaging API（官方帳號 bot 推播）。

```
攝影機 ──▶ 人物偵測 (HOG / YOLO ONNX) ──▶ ROI 判斷 ──▶ 去抖動 + 冷卻 ──▶ LINE push
                                                         └──▶ 存標註截圖
```

## 專案結構

| 路徑 | 說明 |
|---|---|
| `kneo_alert/app.py` | 主迴圈：讀影格 → 偵測 → ROI → 觸發 → 通知；攝影機斷線自動重連 |
| `kneo_alert/detector.py` | 偵測器：`hog`（免模型）、`yolo_onnx`（OpenCV DNN 跑 YOLOv8/YOLO11） |
| `kneo_alert/roi.py` | 多邊形 ROI，`foot`（腳點）或 `overlap`（重疊比例）判斷 |
| `kneo_alert/trigger.py` | 狀態機：連續 N 張有人才算進入、人站著不動只通知一次、冷卻時間 |
| `kneo_alert/notifier.py` | LINE push（含重試與 `X-Line-Retry-Key` 防重複）、背景執行緒送出 |
| `tools/pick_roi.py` | 在畫面上點選 ROI 多邊形，寫回 `config.yaml` |
| `tools/line_webhook_id.py` | 取得群組 ID 用的小型 webhook 伺服器 |
| `tools/send_test_message.py` | 傳一則測試訊息，確認 LINE 設定正確 |
| `deploy/kneo-roi-alert.service` | systemd 開機自動啟動 |

## 1. 安裝

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp config.example.yaml config.yaml
cp .env.example .env
```

> OpenCV 請用 4.x（`requirements.txt` 已鎖定）。OpenCV 5 把 HOG 人物偵測器移出主套件了。
> 板子上如果 pip 沒有現成的 opencv-python wheel，可改用系統套件 `sudo apt install python3-opencv`，建 venv 時加 `--system-site-packages`。

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
     1. Messaging API 分頁 → 允許 bot 加入群組（Allow bot to join group chats，在 LINE Official Account Manager 設定）。
     2. `python tools/line_webhook_id.py --port 8000`
     3. 另開終端機用 `ngrok http 8000`（或 `cloudflared tunnel --url http://localhost:8000`）取得 HTTPS 網址。
     4. Webhook URL 填 `https://<網址>/callback`，打開 **Use webhook**。
     5. 把 bot 邀進群組或在群組裡說話，終端機會印出 `LINE_TO=C...`。
     6. 取得後可以關掉 webhook 與 ngrok，推播不需要 webhook。
4. 測試：`python tools/send_test_message.py`

**免費方案每月推播則數有上限**（傳到群組時每個成員都算一則），請用 `trigger.cooldown_sec` 控制頻率。

## 3. 設定 ROI

```bash
python tools/pick_roi.py            # 需要螢幕（或 SSH -X / VNC）
```

左鍵加點、右鍵或 `u` 復原、`c` 清除、`s` 儲存。座標以 `config.yaml` 的 `camera.width x height` 為準，**改解析度後要重新點選**。

沒有螢幕的話：在 Pi 上存一張畫面，拿到電腦上點選：

```bash
python tools/pick_roi.py --source frame.jpg
```

## 4. 執行

```bash
python -m kneo_alert --dry-run --show   # 先不傳 LINE，看預覽視窗確認偵測與 ROI
python -m kneo_alert                    # 正式執行
python -m kneo_alert --source video.mp4 --dry-run   # 用錄好的影片測試
```

板子沒接螢幕時，用網頁看即時畫面（電腦或手機瀏覽器開 `http://<板子IP>:8080/`）：

```bash
python -m kneo_alert --dry-run --stream          # 預設 8080 埠；--stream 9000 可換埠
```

> 串流沒有密碼保護，只在信任的區域網路使用，不要把這個埠開到外網。沒人在看時不會額外耗 CPU。

預覽視窗／網頁中：黃框是 ROI，紅框是 ROI 內的人，綠框是 ROI 外的人，圓點是判斷用的腳點。

## 4.5 部署到 Kneo Pi

**(1) 在電腦（PowerShell）打包並複製過去**，`<user>@<ip>` 換成板子的帳號與 IP：

```powershell
tar --exclude=.venv --exclude=snapshots --exclude=__pycache__ --exclude=.pytest_cache -czf ..\kneo.tgz .
scp ..\kneo.tgz <user>@<ip>:~/
ssh <user>@<ip> "mkdir -p ~/kneo-roi-line-alert && tar -xzf ~/kneo.tgz -C ~/kneo-roi-line-alert"
```

（`.env`、`config.yaml`、`models/yolov8n.onnx` 會一起帶過去。）

**(2) 在板子上安裝並檢查**：

```bash
ssh <user>@<ip>
cd ~/kneo-roi-line-alert
bash deploy/setup_pi.sh
```

腳本會建 venv、裝套件（沒螢幕時自動改裝 headless 版 OpenCV）、測 YOLO 每張耗時、讀一張攝影機畫面存成 `camera_test.jpg`。

**(3) 重新畫 ROI**：板子的攝影機角度、解析度和電腦不同，ROI 一定要重畫。板子沒螢幕的話：

```powershell
scp <user>@<ip>:~/kneo-roi-line-alert/camera_test.jpg .
.venv\Scripts\python tools\pick_roi.py --source camera_test.jpg   # 存到電腦的 config.yaml
scp config.yaml <user>@<ip>:~/kneo-roi-line-alert/
```

**(4) 測試，再設成開機自動啟動**：

```bash
.venv/bin/python -m kneo_alert --dry-run     # 走進 ROI 看有沒有 ALERT
bash deploy/setup_pi.sh --install-service    # 開機自動啟動、當掉自動重啟
journalctl -u kneo-roi-alert -f              # 看即時 log
```

之後改了 `config.yaml` 要 `sudo systemctl restart kneo-roi-alert` 才會生效。

## 4.6 修改偵測範圍（ROI）——直接在板子上用瀏覽器改

用 `--stream` 執行時，瀏覽器開 `http://<板子IP>:8080/roi`：

1. 畫面會顯示攝影機目前的影像和現在的 ROI（黃色）。
2. 按「清除」，在畫面上依序點出新的範圍（至少 3 點；手機也能點）。
3. 按「儲存」→ **立即生效**，同時寫回板子上的 `config.yaml`，重開也保留。

不用再把 `config.yaml` 在電腦和板子之間傳來傳去。（`tools/pick_roi.py` 仍可在有螢幕的電腦上用。）

## 4.7 多人進入：同時進、陸續進都算

第一個人進入 ROI 後，系統會繼續算 `trigger.gather_sec` 秒（預設 5），這段時間內
**每一波進來的人都會加總**，時間到才傳**一則**「共 N 人」。

| 情況 | 通知 | 截圖（`attach_image: true`） |
|---|---|---|
| 兩人一起進來 | 2 人 | 1 張（兩人都在） |
| 窄門：一人進、離開框，另一人再進 | 2 人 | 2 張 |
| 兩人一起進，第三人之後才到 | 3 人 | 2 張 |
| 一人在框內時另一人加入 | 2 人 | 1 張（換成兩人都在的那張） |

- 文字 + 最多 4 張截圖在同一次推播送出，只算 1 則額度。
- 系統只會數人頭、不認得是誰：同一人在這幾秒內出框又進框會被算兩次；
  前一人離開的同一瞬間後一人補上同一位置（中間完全沒空檔）會被算成 1 人。
- 兩人前後重疊（擋住）時偵測器可能只框到 1 人。
- 窄門人與人間隔較久，可把 `gather_sec` 調大（例如 8～10）；`gather_sec: 0` = 一有人就立刻通知。

## 4.8 通知附上即時截圖（imgbb）

LINE 傳圖片必須是公開的 HTTPS 網址，所以截圖會先上傳到 [imgbb](https://imgbb.com/) 再傳到 LINE。
每一波進來的人一張（最多 4 張），和文字在同一次推播送出，計為 1 則。

1. 到 https://imgbb.com/ 註冊並登入，再到 https://api.imgbb.com/ 取得 API key。
2. 在板子的 `.env` 加上 `IMGBB_API_KEY=你的key`。
3. `config.yaml` 設定 `alert.attach_image: true`；`image_expiration_sec` 是 imgbb 自動刪除時間（預設 7 天）。

注意：

- 有人的照片會上傳到第三方網站（網址很難猜，但知道網址的人都看得到）。
- 上傳失敗時仍會傳文字通知，不會漏掉告警。
- `--dry-run` 不會上傳，只會在 log 顯示「snapshot would be attached」。

## 5. YOLO 模型（建議，比 HOG 準很多）

HOG 不需要模型、可以直接跑，但誤判與漏判較多，正式使用建議換 YOLO。在電腦上匯出 ONNX 再複製到 Pi：

```bash
pip install ultralytics
yolo export model=yolov8n.pt format=onnx imgsz=640 opset=12
# 產生 yolov8n.onnx → 放到 Pi 的 models/yolov8n.onnx
```

`config.yaml` 改成 `detector.type: yolo_onnx`。Pi 的 CPU 跑 640 可能偏慢，可用 `imgsz=320` 匯出並把 `input_size` 改成 320，或調大 `process_every_n_frames`。

### 使用板子的 NPU

如果 Kneo Pi 有 NPU 與廠商 SDK，可以在 `kneo_alert/detector.py` 新增一個繼承 `PersonDetector` 的類別，實作 `detect(frame) -> list[Detection]`（座標為原始影格像素），再在 `create_detector` 註冊新的 `type` 即可，其他部分不需要改。

## 6. 參數調整

| 現象 | 調整 |
|---|---|
| 沒人也通知（誤判） | 提高 `conf_threshold`、`enter_frames`；改用 YOLO |
| 有人卻沒通知 | 降低 `conf_threshold`；確認 ROI 有包含腳的位置（`foot` 模式看的是腳）；或改用 `mode: overlap` |
| 同一個人通知太多次 | 提高 `exit_frames`、`cooldown_sec` |
| CPU 太高 / 太慢 | 降解析度、調大 `process_every_n_frames`、YOLO 用較小的 `input_size` |

## 測試

```bash
pip install -r requirements-dev.txt
pytest
```

## 已知限制

- 目前只傳文字訊息。Messaging API 傳圖片需要公開的 HTTPS 圖片網址（不能直接上傳檔案），所以截圖只存在本機 `snapshots/`。若要傳圖，需要另外架圖床或物件儲存（例如 S3 / Cloudflare R2）。
- 沒有做多目標追蹤；「同一個人只通知一次」是靠 ROI 有人/沒人的狀態切換加上冷卻時間。
