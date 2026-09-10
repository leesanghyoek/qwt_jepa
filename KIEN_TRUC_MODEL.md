# Kiến trúc QWT–JEPA: dữ liệu đi qua model như thế nào

Mọi con số dưới đây lấy trực tiếp từ code với `configs/base.yaml`, không phải ước lượng.

---

## 1. Toàn cảnh một bước train

```
     ẢNH NHIỄU  [B,3,256,256]          ẢNH SẠCH  [B,3,256,256]
     IMU NHIỄU  [B,128,6]              IMU SẠCH  [B,128,6]
          │                                   │
          │  (đầu vào của model)              │  (chỉ dùng làm ĐÁP ÁN)
          ▼                                   ▼
      ┌───────┐                           ┌───────┐
      │  QWT  │                           │  QWT  │
      └───┬───┘                           └───┬───┘
          ▼                                   ▼
   512 token hệ số                     512 token hệ số
          │                                   │
          ▼                                   ▼
   ┌─────────────┐                    ┌──────────────────┐
   │  tokenizer  │                    │ target_tokenizer │ ← bản EMA, đóng băng
   └──────┬──────┘                    └────────┬─────────┘
          │  [B,512,384]                       │  [B,512,384]
          ▼                                    ▼
   ╔═══════════════╗                   ┌────────────────┐
   ║   MẶT NẠ      ║                   │ target_encoder │ ← bản EMA, đóng băng
   ║ che 41% token ║                   └────────┬───────┘
   ╚═══════╤═══════╝                            │
          │ chỉ còn 152 token                   ▼
          ▼                                  z_tgt  ────────┐
   ┌─────────────────┐                    (đáp án latent)   │
   │ context_encoder │  12 lớp, 21.29M                      │
   └────────┬────────┘                                      │
           │ z_ctx  [B,152,384]                             │
           ├──────────────────────┐                         │
           ▼                      │                         │
     ┌───────────┐                │                         │
     │ predictor │  2.03M         │                         │
     └─────┬─────┘                │                         │
          │ z_pred [B,208,384]    │                         │
          │                       │                         ▼
          │                       │                   ┌───────────┐
          ├───────────────────────┴──────────────────►│  L_jepa   │
          │                                            └───────────┘
          ▼
   ┌──────────────────┐
   │  ghép emb_full   │  z_ctx + z_pred + missing_token
   └────────┬─────────┘
           │ [B,512,384]
      ┌────┴─────┐
      ▼          ▼
 ┌─────────┐ ┌────────┐
 │image_head│ │imu_head│
 └────┬────┘ └───┬────┘
      ▼          ▼
  ảnh dựng    IMU dựng  ──────────► L_img , L_imu  (so với bản SẠCH)
```

---

## 2. Từ ảnh/IMU thành 512 token

QWT tách tín hiệu thành các **dải tần số**. Mỗi dải cắt thành các mảnh, mỗi mảnh = 1 token.

### Token ảnh (256 token)

| Dải | Lưới | Token | Vị trí | Nội dung |
|---|---|---:|---|---|
| L1 LH | 8×8 | 64 | 0–63 | chi tiết mịn nhất |
| L1 HL | 8×8 | 64 | 64–127 | chi tiết mịn nhất |
| L1 HH | 8×8 | 64 | 128–191 | chi tiết mịn nhất |
| L2 LH | 4×4 | 16 | 192–207 | chi tiết vừa |
| L2 HL | 4×4 | 16 | 208–223 | chi tiết vừa |
| L2 HH | 4×4 | 16 | 224–239 | chi tiết vừa |
| **L3 LL** | 2×2 | **4** | **240–243** | **ảnh thu nhỏ 32×32 — 94% năng lượng** |
| L3 LH | 2×2 | 4 | 244–247 | chi tiết thô |
| L3 HL | 2×2 | 4 | 248–251 | chi tiết thô |
| L3 HH | 2×2 | 4 | 252–255 | chi tiết thô |

### Token IMU (256 token)

| Nhóm | Dải | Token | Vị trí |
|---|---|---:|---|
| acc | L1 D | 64 | 256–319 |
| acc | L2 D | 32 | 320–351 |
| **acc** | **L3 A** | **16** | **352–367** |
| acc | L3 D | 16 | 368–383 |
| gyro | L1 D | 64 | 384–447 |
| gyro | L2 D | 32 | 448–479 |
| **gyro** | **L3 A** | **16** | **480–495** |
| gyro | L3 D | 16 | 496–511 |

Hai dải in đậm (`L3 LL` của ảnh, `L3 A` của IMU) = **36 token** mang gần hết nội dung.
Chúng được `masking.protect_coarse` bảo vệ, không bao giờ bị che.

### Tokenizer biến hệ số thành vector 384 chiều

```
hệ số 1 dải  ──► chia cho band_scale ──► Linear ──► + pos_embed
                 (cân bằng biên độ)      384 chiều   + modality_emb
                                                     + scale_emb
                                                          │
                                                          ▼
                                                   token [384]
```

---

## 3. Mặt nạ chia 512 token làm ba phần

Đo thật trên 500 lần bốc:

```
┌──────────────────────── 512 token ────────────────────────┐
│                                                            │
│  CONTEXT 152 (30%)   TARGET 208 (41%)    VỨT 152 (30%)    │
│  ───────────────     ───────────────     ─────────────    │
│  dữ liệu THẬT        model phải đoán     thay bằng        │
│  (bản NHIỄU)         (không được xem)    missing_token    │
│                                          — 1 hằng số       │
│  → vào encoder       → predictor đoán    → không mang      │
│                                            thông tin gì    │
└────────────────────────────────────────────────────────────┘
```

---

## 4. Ba nhánh, và cái nào dùng lúc nào

| Module | Tham số | Train | Triển khai |
|---|---:|:---:|:---:|
| `tokenizer` | 0.50M | ✓ | ✓ |
| `context_encoder` | 21.29M | ✓ | ✓ |
| `image_head` | 0.30M | ✓ | ✓ (nếu muốn ảnh sạch) |
| `imu_head` | 0.004M | ✓ | ✓ (nếu muốn IMU sạch) |
| `predictor` | 2.03M | ✓ | ✗ |
| `target_tokenizer` | 0.50M | đóng băng (EMA) | ✗ |
| `target_encoder` | 21.29M | đóng băng (EMA) | ✗ |
| `missing_token` | 0.0004M | ✓ | ✗ |
| **Tổng** | **45.92M** | 24.12M có gradient | **22.1M** |

Nhánh target là **bản sao EMA** của nhánh context: nó không học bằng gradient, mà mỗi
bước bị kéo nhích về phía bản chính: `target = 0.996·target + 0.004·context`.
Nó tồn tại để tạo ra "đáp án" `z_tgt` ổn định cho `L_jepa`.

---

## 5. Bốn loss

```
L_jepa = smooth_l1( z_pred , z_tgt )        dự đoán latent phần bị che
L_img  = charbonnier( ảnh dựng , ảnh sạch ) khử nhiễu ảnh
L_imu  = L1(acc) + L1(gyro)                 khử nhiễu IMU
L_var  = relu( 1.0 − độ đa dạng của z_ctx ) chống collapse

L = L_jepa + 1.0·L_img + 1.0·L_imu + 1.0·L_var
```

---

## 6. Vấn đề hiện tại

Nhìn lại mục 3 rồi nhìn mục 1: **recon head ăn `emb_full`**, mà `emb_full` là:

```
   152 token thật (nhưng là bản NHIỄU)     30%
 + 208 token do predictor đoán             41%   ← đo được là VÔ DỤNG
 + 152 token hằng số missing_token         30%   ← không có thông tin
```

Tỉ số `L_jepa / pos-only` ổn định ở **1.1** nghĩa là predictor còn thua cái mẹo
"bỏ qua ảnh, đoán theo vị trí token". Nên 41% kia là rác.

→ Head thực chất chỉ có **30% thông tin thật** để dựng lại toàn bộ ảnh. Kết quả đo:

| | Model | Chỉ copy ảnh nhiễu |
|---|---|---|
| PSNR | 17.65 dB | **21.99 dB** |
| RMSE acc | 0.285 | **0.123** |

Thêm một lỗi nữa: `context_encoder` **lúc train chưa bao giờ thấy đủ 512 token**
(luôn chỉ ~152). Nhưng lúc triển khai (`forward_full`) lại nạp cả 512 → chạy ngoài
phân bố nó được huấn luyện.

---

## 7. Đề xuất: hai LƯỢT CHẠY, vẫn MỘT model

Đây là chỗ dễ hiểu nhầm nhất. **Không tạo thêm model nào.** Vẫn đúng những module ở
mục 4, đúng 24.12M tham số. Chỉ là dữ liệu chạy qua chúng **hai lần** mỗi bước train.

Giống một học sinh làm hai bài tập bằng cùng một bộ não — không phải thuê thêm học sinh.

```
                    tokenizer          ◄── MỘT module
                        │
          ┌─────────────┴─────────────┐
          │                           │
     LƯỢT 1: CHE                 LƯỢT 2: ĐỦ
     152 token                   cả 512 token
          │                           │
          ▼                           ▼
   context_encoder  ◄══════════ context_encoder
          │          CÙNG MỘT MODULE         │
          │          (gọi 2 lần)             │
          ▼                                  ▼
     predictor                        image_head / imu_head
          │                                  │
          ▼                                  ▼
       L_jepa                          L_img , L_imu
```

Trong code, khác biệt chỉ là mấy dòng:

```python
tok = self.tokenizer(q_img_n, q_imu_n)              # 1 lần, dùng chung

# LƯỢT 1 — che, để học dự đoán
z_ctx  = self.context_encoder(tok[:, mask.context_index])   # ~152 token
z_pred = self.predictor(z_ctx, ...)                        # → L_jepa

# LƯỢT 2 — đủ, để học khử nhiễu
z_all   = self.context_encoder(tok)                        # cả 512 token
img_rec = self.image_head(z_all, ...)                      # → L_img
imu_rec = self.imu_head(z_all, ...)                        # → L_imu
```

`self.context_encoder` xuất hiện hai lần = **một module được gọi hai lần**, y như gọi
cùng một hàm với hai tham số khác nhau. Gradient của cả hai lượt cộng dồn vào **cùng
một bộ trọng số** — đó chính là điều ta muốn: encoder học từ cả hai nhiệm vụ.

### Vì sao không gộp làm một lượt

```python
z_all = self.context_encoder(tok)        # mọi token đã nhìn thấy nhau qua attention
z_ctx = z_all[:, mask.context_index]     # ← ĐÃ RÒ RỈ
z_pred = self.predictor(z_ctx, ...)      # đoán cái nó đã biết → L_jepa vô nghĩa
```

Transformer có attention toàn cục: sau một lượt, token context đã trộn thông tin của
token target. Bài toán "đoán phần bị che" thành gian lận.

### Được gì

- Head nhận **đủ 512 token nhiễu** → bài toán khử nhiễu thật, có cơ hội vượt 21.99 dB
- Encoder được train đúng chế độ sẽ dùng lúc triển khai → hết lệch train/test
- `L_jepa` giữ nguyên mục tiêu che, không rò rỉ

### Mất gì

Encoder chạy 2 lần thay vì 1. Throughput ~60 → ~38 im/s. Một run 40 epoch từ 3,5h
lên ~5,5h — vẫn gọn trong một phiên Kaggle 12h.

Số tham số **không đổi**.
