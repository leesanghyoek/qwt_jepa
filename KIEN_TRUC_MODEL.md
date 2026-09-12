# Kiến trúc QWT–JEPA: kích thước và công thức

Mọi con số lấy trực tiếp từ code với `qwt_jepa/configs/base.yaml`.
Ý nghĩa thiết kế và bằng chứng đo đạc: `MODEL_LA_GI.md`.
Cách đọc log: `LOG_TRAIN_GIAI_THICH.md`.

---

## 1. Siêu tham số

| | |
|---|---|
| `d_model` | 384 |
| encoder | 12 lớp, 6 head, `mlp_ratio` 4.0, pre-norm |
| predictor | `d_pred` 192, 4 lớp, 6 head |
| ảnh vào | 256×256 RGB |
| IMU vào | 128 mẫu × 6 kênh (acc 3 + gyro 3), 100 Hz |
| QWT | 3 mức cho cả ảnh và IMU |
| patch | 16 |
| tổng token | **512** (256 ảnh + 256 IMU) |

---

## 2. Tham số

| Module | Tham số | Vai trò |
|---|---:|---|
| `tokenizer` | 0.502M | hệ số QWT → vector 384 chiều |
| `context_encoder` | 21.294M | 12 lớp transformer |
| `image_head` | 1.771M | latent → hệ số ảnh |
| `imu_head` | 0.002M | latent → hệ số IMU |
| `predictor` | 2.026M | đoán latent phần bị che |
| `target_tokenizer` | 0.502M | bản EMA, **đóng băng** |
| `target_encoder` | 21.294M | bản EMA, **đóng băng** |
| `missing_token` | 0.0004M | đóng băng khi `recon_from_full` |
| **Tổng** | **47.39M** | |
| **Học (GĐ1)** | **25.60M** | trừ nhánh target EMA |
| **Học (GĐ2)** | **1.77M** | chỉ 2 recon head |
| **Triển khai** | **23.57M** | tokenizer + encoder + 2 head |

---

## 3. Từ tín hiệu đến 512 token

### 3.1 QWT

```
ảnh  [B, 3, 256, 256] ──rgb_to_quat──► [B, 4, 256, 256] ──image_qwt(3)──► kim tự tháp
IMU  [B, 128, 6]      ────────────────────────────────── imu_qwt(3)  ──► kim tự tháp
```

Chỉ dùng **phần vector (i, j, k)** của hệ số quaternion; phần thực `w` luôn ≈ 0 với Haar.

### 3.2 Token ảnh — 256 token

Mỗi dải cắt thành lưới patch 16×16.

| Dải | Kích thước dải | Lưới | Token | Vị trí | Nội dung |
|---|---|---|---:|---|---|
| L1 LH | 128×128 | 8×8 | 64 | 0–63 | chi tiết mịn nhất |
| L1 HL | 128×128 | 8×8 | 64 | 64–127 | " |
| L1 HH | 128×128 | 8×8 | 64 | 128–191 | " |
| L2 LH | 64×64 | 4×4 | 16 | 192–207 | chi tiết vừa |
| L2 HL | 64×64 | 4×4 | 16 | 208–223 | " |
| L2 HH | 64×64 | 4×4 | 16 | 224–239 | " |
| **L3 LL** | 32×32 | 2×2 | **4** | **240–243** | **ảnh thu nhỏ — 94% năng lượng** |
| L3 LH | 32×32 | 2×2 | 4 | 244–247 | chi tiết thô |
| L3 HL | 32×32 | 2×2 | 4 | 248–251 | " |
| L3 HH | 32×32 | 2×2 | 4 | 252–255 | " |

### 3.3 Token IMU — 256 token

Mỗi bước thời gian của mỗi dải là một token.

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

Hai dải in đậm (`L3 LL` ảnh, `L3 A` IMU) = **36 token**, được `protect_coarse` bảo vệ.

### 3.4 Tokenizer

Cho dải thứ `i`, patch hoá rồi chiếu:

```
ảnh:  hệ số [B, 3, h, w] ──/ band_scale[i]──► patchify ──► [B, n, 768] ──Linear(768→384)──► [B, n, 384]
IMU:  hệ số [B, L, 3]    ──/ band_scale[i]──►                          ──Linear(3→384)  ──► [B, L, 384]
```

Rồi cộng ba embedding học được:

```
token = proj(hệ số / band_scale) + pos_embed[512,384] + modality_emb[3,384] + scale_emb[18,384]
```

`band_scale` = std của hệ số dải đó, đo trên 8 batch train. Không có nó, biên độ chênh
**123×** giữa các dải và encoder gần như không thấy chi tiết mịn.

**Ra: `[B, 512, 384]`**

---

## 4. Encoder

```
x → 12 × TransformerBlock → LayerNorm
```

Mỗi block (pre-norm):
```
x = x + MHA(LayerNorm(x))                       384 chiều, 6 head
x = x + MLP(LayerNorm(x))                       384 → 1536 → GELU → 384
```

Tham số mỗi block: attn `in_proj` (1152,384) + `out_proj` (384,384) + mlp (1536,384) + (384,1536) ≈ 1.77M. × 12 = 21.29M.

---

## 5. Mặt nạ (chỉ dùng ở giai đoạn 1)

Đo thật trên 500 lần bốc:

```
┌──────────────────── 512 token ────────────────────┐
│ CONTEXT 152 (30%)  TARGET 208 (41%)  VỨT 152 (30%)│
│ dữ liệu THẬT       model phải đoán   missing_token │
│ (bản NHIỄU)                          (hằng số)     │
└────────────────────────────────────────────────────┘
```

Cách chọn: bốc 4 hình chữ nhật (diện tích 15–25%, tỉ lệ 0.75–1.5) trên lưới chuẩn hoá
`[0,1]²`, áp **cùng một hình** lên mọi dải ảnh; 2 đoạn thời gian (15–25%) cho IMU.
Token dải thô bị loại khỏi target và luôn giữ trong context.

---

## 6. Predictor

```
z_ctx [B,152,384] ──Linear(384→192)──► + pos_embed[ctx_index]  ┐
                                                                ├─ nối ─► [B,360,192]
mask_token [1,1,192] ──expand──► + pos_embed[tgt_index]        ┘
                            │
                    4 × TransformerBlock (192 chiều, 6 head)
                            │
                        LayerNorm
                            │
              lấy 208 vị trí target ──Linear(192→384)──► z_pred [B,208,384]
```

---

## 7. Hai recon head

Cho dải thứ `i`, với `raw` = hệ số nhiễu đã chuẩn hoá:

```
tok   = cat([ emb[:, start:end],  patchify(raw) ])          # [B, n, 384+768]
g, d  = Linear(1152 → 1536)(tok).chunk(2)                   # mỗi cái [B, n, 768]
gain  = 1 + gate_max · tanh(g)                              # ∈ [0, 2]
norm  = raw · gain + d
hệ số = norm · band_scale[i]
```

IMU y hệt với `Linear(384+3 → 6)`.

Rồi ghép lại thành kim tự tháp và biến đổi ngược:

```
image_iqwt → [B, 4, 256, 256] → lấy kênh 1:4 → ảnh [B, 3, 256, 256]
imu_iqwt   → IMU [B, 128, 6]
```

**Ba tính chất:**
- `proj` khởi tạo **bằng 0** → `g = d = 0` → `gain = 1` → **đầu ra = đầu vào chính xác**
- `tanh` chặn `gain ∈ [0, 2]`: đủ cho co hệ số (<1) và khử nhoè (>1), không nổ được
- `skip` mang **hệ số thô 768 chiều**, không phải đầu ra tokenizer (đã nén 2:1)

---

## 8. Công thức loss

### `L_jepa` — dự đoán latent

```
L_jepa = SmoothL1( z_pred , stop_grad(z_tgt) )

                    ⎧ 0.5·d²        nếu |d| < 1
SmoothL1(d) = mean ⎨                                  ,  d = z_pred − z_tgt
                    ⎩ |d| − 0.5     nếu |d| ≥ 1
```

`z_tgt` tính từ dữ liệu **sạch** qua nhánh EMA, có `detach()` nên không có gradient.

### `L_img` — tái tạo ảnh (Charbonnier)

```
L_img = mean √( (img_rec − img_clean)² + ε² )        ε = 1e-3
```

Charbonnier ≈ L1 nhưng khả vi tại 0. Ảnh thang `[0, 1]`.

### `L_imu` — tái tạo IMU

```
L_imu = L1( rec[...,0:3], clean[...,0:3] )  +  L1( rec[...,3:6], clean[...,3:6] )
        └────────── acc ──────────┘            └────────── gyro ──────────┘
```

Là **tổng hai số**. Đơn vị là **số lần std** (IMU đã chuẩn hoá z-score).

### `L_var` — chống collapse (hinge kiểu VICReg)

```
std[t,d] = √( var_theo_batch( z_ctx[:, t, d] ) + 1e-4 )      # [512, 384]

L_var = mean( relu( γ − std ) )                               γ = var_gamma = 1.0
```

**Tính phương sai theo trục BATCH** (đổi ảnh thì biểu diễn đổi bao nhiêu), không phải
gộp cả (batch, token). Chỉ áp cho `z_ctx`, **không** cho `z_pred` — predictor kết thúc
bằng `Linear` không chuẩn hoá nên sẽ lách bằng cách phóng to scale.

### `L_band` — cân bằng dải (mặc định TẮT)

```
L_band = mean qua các dải [ Charbonnier( hệ_số_dự_đoán_chuẩn_hoá , hệ_số_sạch_chuẩn_hoá ) ]
```

Đo A/B cho thấy **làm tệ hơn** → `lambda_band = 0.0`.

### Tổng

```
L = λ_jepa·L_jepa + λ_img·L_img + λ_imu·L_imu + λ_band·L_band + λ_var·L_var
```

| | GĐ1 | GĐ2 |
|---|---|---|
| `λ_jepa` | 1.0 | **0.0** |
| `λ_img` | **0.0** | 1.0 |
| `λ_imu` | **0.0** | 1.0 |
| `λ_var` | 1.0 | **0.0** |
| `λ_band` | 0.0 | 0.0 |

---

## 9. Chỉ số theo dõi (không phải loss, không có gradient)

```
sharpness(x)  = mean|x[:,:,1:,:] − x[:,:,:-1,:]|  +  mean|x[:,:,:,1:] − x[:,:,:,:-1]|
net           = sharpness(img_rec) / sharpness(img_clean)        1.0 = nét bằng ảnh gốc

content_std(z) = mean( √( var_theo_batch(z) + 1e-4 ) )           đổi ảnh → biểu diễn đổi bao nhiêu

L_jepa_pos    = SmoothL1( mean_theo_batch(z_tgt) broadcast , z_tgt )
                ↑ mốc "gian lận": đoán theo vị trí token, bỏ qua ảnh.
                  L_jepa / L_jepa_pos phải < 1.0

PSNR = −10·log₁₀( mean( (img_rec − img_clean)² ) )               MAX = 1.0
RMSE = √( mean( (imu_rec − imu_clean)² ) )                       đơn vị: số lần std
```

---

## 10. Luồng dữ liệu đầy đủ

### Giai đoạn 1 — tiền huấn luyện

```
ảnh nhiễu [B,3,256,256] ┐                    ảnh sạch [B,3,256,256] ┐
IMU nhiễu [B,128,6]     ┘                    IMU sạch [B,128,6]     ┘
        │ QWT + tokenizer                            │ QWT + target_tokenizer (EMA)
        ▼                                            ▼
   [B,512,384]                                  [B,512,384]
        │ mask → giữ 152                             │ target_encoder (EMA)
        ▼                                            ▼
   [B,152,384] ──context_encoder──► z_ctx       [B,512,384]
        │                                            │ lấy 208 vị trí target
        ▼ predictor                                  ▼
   z_pred [B,208,384] ─────────► L_jepa ◄──── z_tgt [B,208,384]
        
   z_ctx ──────────────────────► L_var
```

### Giai đoạn 2 — tinh chỉnh phục hồi

```
ảnh nhiễu [B,3,256,256] + IMU nhiễu [B,128,6]
        │ QWT + tokenizer ❄️
        ▼
   [B,512,384]   ← KHÔNG che
        │ context_encoder ❄️
        ▼
   [B,512,384]
        ├──► image_head ──iQWT──► ảnh [B,3,256,256] ──► L_img
        └──► imu_head   ──iQWT──► IMU [B,128,6]     ──► L_imu

❄️ = đóng băng (requires_grad = False)
```

---

## 11. Kích thước vào/ra tóm tắt

| Giai đoạn | Vào | Ra | Loss so với |
|---|---|---|---|
| **1** | ảnh+IMU nhiễu **và** ảnh+IMU sạch | `z_pred [B,208,384]` | `z_tgt [B,208,384]` |
| **2** | ảnh+IMU **nhiễu** | ảnh `[B,3,256,256]`<br>IMU `[B,128,6]` | ảnh+IMU **sạch** |
| **Triển khai** | ảnh+IMU nhiễu | ảnh sạch + IMU sạch | — |

Đường triển khai duy nhất là `QwtJepa.reconstruct(img_noisy, imu_noisy)`. Đừng tự ghép
lại — nó phải khớp chính xác với nhánh tái tạo lúc train, kể cả skip connection.
