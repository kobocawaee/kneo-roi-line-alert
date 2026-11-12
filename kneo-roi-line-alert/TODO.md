# 待辦（webcam 版）

> Xtion 版在隔壁的 `kneo-roi-line-alert-xtion` 資料夾。

## ⏸ 暫緩：開機自動啟動（systemd）

**為什麼暫緩**（2026-11-12）：LINE 免費方案每月只有 200 則，還有東西要測試，
服務一直跑會在測試期間持續消耗額度。測試完、確定設定後再裝。

**要做的時候**（在板子上）：

```bash
cd ~/kneo-roi-line-alert
bash deploy/setup_pi.sh --install-service
# 要順便開網頁即時畫面：
sed -i 's#-m kneo_alert$#-m kneo_alert --stream#' /etc/systemd/system/kneo-roi-alert.service
systemctl daemon-reload && systemctl restart kneo-roi-alert
reboot   # 重開後用瀏覽器 http://<IP>:8080/ 和走進 ROI 驗證
```

常用：`systemctl status kneo-roi-alert`、`journalctl -u kneo-roi-alert -f`、
`systemctl stop|disable kneo-roi-alert`。

## ⚠️ 交作業前

- **刪掉 `.env`**（裡面有 LINE token、imgbb API key），只交 `.env.example`。
- 不要交 `config.yaml` 以外的本機產物：`.venv/`、`snapshots/`（有人臉照片）、`camera_test.jpg`。
- 壓縮前確認：`tar --exclude=.env --exclude=.venv --exclude=snapshots ...`

## 測試期間省額度的做法

- 一律加 `--dry-run`（只印訊息、不傳 LINE），只有要驗證 LINE 本身時才拿掉。
- `trigger.cooldown_sec` 目前 60 秒；測試時可暫時調大。
- 本月用量：LINE Official Account Manager →「設定」。

## 目前狀態（2026-11-12）

- [x] 電腦端：webcam + YOLO（320）+ ROI + LINE 推播測通
- [x] Kneo Pi（Arch Linux ARM, aarch64, Python 3.14）：環境建置
- [x] 多人進入：同時進、陸續進都算，傳一則總人數（`gather_sec`）— 板子上測過
- [x] 通知附截圖（imgbb；每波進來的人一張、最多 4 張）
- [x] 瀏覽器修改 ROI（`http://<IP>:8080/roi`）
- [x] 板子接 USB webcam（`source: 8`）實測
- [x] 板子上實際傳 LINE 文字 + 截圖成功（2026-11-12）
- [x] 教授要求 1：模型轉 NEF，在 KL730 NPU 上執行（每張約 55 ms；用家裡 100 張畫面校正）
- [ ] 教授要求 2：人臉偵測模型（已編進同一個 NEF；`require_face` 尚待板子實測）
- [ ] 教授要求 3：雙向傳輸（LINE 下指令、板子回覆；需要 webhook + 通道）
- [ ] 板子搬回實驗室後：用實驗室畫面重拍校正圖片、重新編譯 NEF
- [ ] 開機自動啟動（上面，暫緩）

## 之後可以做

- 傳到教授的群組（需 group ID；每個成員都算一則）
- 排線鏡頭 UC-439：板子只支援 IMX219，開機偵測失敗；需先確認感光元件型號
