# 待辦（Xtion 版）

> 交付用的 webcam 版在隔壁的 `kneo-roi-line-alert` 資料夾。共用程式由 `../sync_versions.py` 從 webcam 版同步過來。

## ⏸ 暫緩：開機自動啟動（systemd）

**為什麼暫緩**（2026-11-12）：LINE 免費方案每月只有 200 則，還有東西要測試，
服務一直跑會在測試期間持續消耗額度。測試完、確定設定後再裝。

**要做的時候**（在板子上）：

```bash
cd ~/kneo-roi-line-alert-xtion
bash deploy/setup_pi.sh --install-service
# 要順便開網頁即時畫面：
sed -i 's#-m kneo_alert$#-m kneo_alert --stream#' /etc/systemd/system/kneo-roi-alert.service
systemctl daemon-reload && systemctl restart kneo-roi-alert
reboot   # 重開後用瀏覽器 http://<IP>:8080/ 和走進 ROI 驗證
```

常用：`systemctl status kneo-roi-alert`、`journalctl -u kneo-roi-alert -f`、
`systemctl stop|disable kneo-roi-alert`。

## 測試期間省額度的做法

- 一律加 `--dry-run`（只印訊息、不傳 LINE），只有要驗證 LINE 本身時才拿掉。
- `trigger.cooldown_sec` 目前 60 秒；測試時可暫時調大。
- 本月用量：LINE Official Account Manager →「設定」。

## 目前狀態（2026-11-12）

- [x] 電腦端：webcam + YOLO（320）+ ROI + LINE 推播測通
- [x] Kneo Pi（Arch Linux ARM, aarch64, Python 3.14）：環境、OpenNI2 編譯安裝
- [x] Xtion PRO Live 透過 OpenNI2 讀彩色影像（640×480）
- [x] 板子上 dry-run 觸發成功、網頁即時畫面（`--stream`）流暢
- [x] 多人先後進入、通知附截圖、瀏覽器修改 ROI（與 webcam 版共用）
- [ ] 板子上實際傳一次 LINE
- [x] 教授要求 1：模型轉 NEF，在 KL730 NPU 上執行（每張約 55 ms；用家裡 100 張畫面校正）
- [ ] 教授要求 2：人臉偵測模型（已編進同一個 NEF；`require_face` 尚待板子實測）
- [ ] 教授要求 3：雙向傳輸（LINE 下指令、板子回覆；需要 webhook + 通道）
- [ ] 板子搬回實驗室後：用實驗室畫面重拍校正圖片、重新編譯 NEF
- [ ] 開機自動啟動（上面，暫緩）

## 之後可以做

- 傳到教授的群組（需 group ID；每個成員都算一則）
- 排線鏡頭 UC-439：板子只支援 IMX219，開機偵測失敗；需先確認感光元件型號
