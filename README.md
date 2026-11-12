# Kneo Pi ROI person-entry LINE alert

Kneo Pi 開發板接攝影機，偵測到有人走進指定區域（ROI）時，透過 **LINE Messaging API** 傳通知（可附即時截圖）。

| 資料夾 | 攝影機 | 說明 |
|---|---|---|
| [`kneo-roi-line-alert/`](kneo-roi-line-alert/) | USB webcam | 主要版本 |
| [`kneo-roi-line-alert-xtion/`](kneo-roi-line-alert-xtion/) | ASUS Xtion PRO Live（OpenNI2） | 含 aarch64 上編譯 OpenNI2 的腳本與 Python 3.14 相容修正 |

兩版共用的程式完全相同，只有攝影機相關檔案不同；新功能先做在 webcam 版，再用
`python sync_versions.py --apply` 同步到 Xtion 版（不同的檔案列在 `sync_versions.py` 的 `VERSION_SPECIFIC`）。

功能：YOLOv8n（ONNX，OpenCV DNN）人物偵測、多邊形 ROI、同時/陸續進入人數統計、
冷卻時間、LINE 文字 + 截圖（imgbb）、瀏覽器即時畫面與 ROI 編輯（`--stream`）。

安裝與使用請看各資料夾的 README。需要自行準備：LINE Messaging API channel、
（選用）imgbb API key、以及自行匯出的 YOLO ONNX 模型（見 README「YOLO 模型」）。
