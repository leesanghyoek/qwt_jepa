# Đọc log train QWT–JEPA: từng giá trị nghĩa là gì, tính trên tập nào, bao nhiêu là tốt

Tài liệu này giải thích **mọi con số** xuất hiện trong log của `python -m qwt_jepa.train.train`,
kèm công thức, vị trí trong code, tập dữ liệu dùng để tính, và **ngưỡng baseline đo thật**
để biết giá trị nào là tốt.

Mọi ngưỡng trong tài liệu đều đo lại được bằng:

```bash
python -m qwt_jepa.scripts.baseline_metrics --split valid --n 128
```

> ⚠️ Ngưỡng phụ thuộc mục `corruption:` trong `qwt_jepa/configs/base.yaml`.
> Sửa mức nhiễu thì **phải chạy lại script trên**, các số dưới đây sẽ đổi.

---

## 1. Bảng tra nhanh

| Chỉ số | Vô dụng | Ngưỡng phải vượt | Tốt | Rất tốt | Cảnh báo |
|---|---|---|---|---|---|
| `z_std` / `ctx` | — | — | 0.5 – 1.0 | ổn định không tụt | 🔴 tụt dần → collapse |
| `L_jepa`/`pos` | ≥ 1.0 | < 1.0 | < 0.5 | < 0.2 | 🔴 ≥ 1.0 = tệ hơn đoán bừa theo vị trí |
| `L_var` | — | — | < 0.1 | ≈ 0 | 🔴 > 0.3 kéo dài = `z_ctx` chưa bung ra |
| `L_jepa` | ≈ 0.42 | < 0.29 | 0.08 – 0.17 | < 0.05 | 🔴 < 0.02 kèm `z_std` tụt |
| `PSNR` | 14.0 dB | **24.5 dB** | 25 – 27 dB | > 28 dB | đứng im khi `L_jepa` giảm |
| `RMSE acc` | 1.0 | **0.115** | < 0.08 | < 0.05 | — |
| `RMSE gyro` | 1.0 | **0.117** | < 0.08 | < 0.05 | — |
| `L_img` | 0.193 | **0.042** | < 0.035 | < 0.025 | — |
| `L_imu` | ≈ 1.6 | **0.171** | < 0.15 | < 0.10 | — |

**Đọc theo thứ tự ưu tiên:** `z_std` → `PSNR` + `RMSE` → `L_jepa`.
Không bao giờ đánh giá model chỉ bằng `L` hoặc chỉ bằng `L_jepa`.

> ⚠️ Các mốc trên ứng với `corruption.image` ở chế độ **NHOÈ CHI PHỐI** (từ commit đổi
> config sang blur-dominant). Mốc của chế độ nhiễu cũ là PSNR 21.99 / `L_img` 0.066 —
> **không so sánh chéo hai chế độ được**. Đổi mức nhiễu thì phải chạy lại
> `scripts/baseline_metrics.py`.

---

## 2. Hai dòng log

### 2.1 Dòng train — `qwt_jepa/train/engine.py:86-95`

```
e5  150/500 step   2650 | L 1.2345 (jepa 0.4210 img 0.6120 imu 0.2015 var 0.0120) | z_std 0.987 ctx 0.981 | lr 1.45e-04 m 0.9966 | 128.3 im/s
```

In ra mỗi `train.log_every: 50` batch.

> **QUAN TRỌNG:** đây là giá trị của **một batch duy nhất** (batch thứ `i`), không phải
> trung bình epoch. Trong `train_one_epoch` không có biến cộng dồn nào. Vì vậy con số này
> dao động ±10–20% là bình thường và **không dùng để kết luận xu hướng hội tụ**.

### 2.2 Dòng eval — `qwt_jepa/train/train.py:179-183`

```
[eval e5] L_jepa 0.0510/0.0569pos | z_std 0.511 | PSNR 10.12 dB | RMSE acc 0.2709 gyro 0.1415
```

Hai con số cần đọc kỹ:

- **`0.0510/0.0569pos`** — `L_jepa` của model / `L_jepa` mà kẻ gian lận **chỉ đoán theo vị trí
  token, không nhìn ảnh** đạt được. Tỉ số phải **< 1.0**; nếu ≥ 1.0 thì JEPA không học được gì
  về nội dung ảnh (xem mục 10).
- **`z_std`** — std của `z_ctx` **theo nội dung**: đổi ảnh đầu vào thì biểu diễn đổi bao nhiêu.
  Không phải std gộp cả (batch, token).

In một lần sau mỗi epoch. Đây là **trung bình có trọng số theo batch size**
(`engine.py:125-131`) trên tối đa `train.limit_val_batches: 100` batch.
Đây mới là số đáng tin để theo dõi.

---

## 3. Dữ liệu được tính trên tập nào?

### 3.1 Nguồn

`qwt_jepa/train/train.py:30-49` (`build_loaders`):

| | Train | Eval |
|---|---|---|
| Manifest | `tartanair-v2-jepa/train/manifest.csv` | `tartanair-v2-jepa/valid/manifest.csv` |
| Ảnh + IMU thật đọc từ | `data.root` = `/home/buidinhkhoi/Datasets/tartanair-v2` | như bên trái |
| Số dòng manifest | 20 904 | 5 286 |
| Số mẫu dùng được | — | **5 118** |
| `shuffle` | `True`, `drop_last=True` | `False`, `drop_last=False` |
| Số batch mỗi epoch | `limit_train_batches: 500` | `limit_val_batches: 100` |

Manifest chỉ chứa **đường dẫn tương đối**; ảnh và IMU nằm ở `data.root`, không phải `jepa_root`.

**168 dòng valid bị loại** tại `data/dataset.py:68-70`: dòng nào có
`imu_end_row_exclusive - 128 < 0` thì bỏ, vì cửa sổ IMU 1.28 s bị hụt ở đầu mỗi trajectory
(khoảng 13 frame đầu của mỗi quỹ đạo).

### 3.2 Cách chia train/valid — có một điểm cần biết

`enforce_env_split: false` trong `base.yaml`, nên `_env_filter` trả về `None`
(`data/dataset.py:87-92`) — **không lọc environment nào cả**. Kết quả thực tế:

```
Trajectory trùng nhau giữa train và valid : 0     ✅
Environment xuất hiện ở cả hai split      : 7/7   ⚠️
```

Phân bố environment:

| Environment | train | valid |
|---|---:|---:|
| AmericanDiner | 1 807 | 607 |
| ArchVizTinyHouseDay | 957 | 238 |
| CountryHouse | 2 349 | 435 |
| DesertGasStation | 3 697 | 991 |
| OldTownNight | 2 000 | 1 405 |
| RetroOffice | 997 | 268 |
| WaterMillDay | 9 097 | 1 342 |

→ Chỉ số val đang đo **khả năng tổng quát hoá sang quỹ đạo mới trong môi trường đã thấy**,
KHÔNG phải sang môi trường mới. Muốn đo cái sau, bật `enforce_env_split: true` — khi đó
valid chỉ còn `OldTownNight` và số liệu sẽ **xấu đi rõ rệt**, đó là điều bình thường.

### 3.3 Nhiễu được sinh như thế nào

TartanAir là ảnh **sạch**. Nhiễu sinh on-the-fly trong `__getitem__` (`data/dataset.py:114-124`):

```python
rng = np.random.default_rng(_stable_seed(env, traj, frame_idx, self.epoch))
img_noisy = corrupt_image(img_clean, rng, ...)
imu_noisy = corrupt_imu(imu_clean, rng, ...)
```

Hệ quả cần nhớ:

- Seed có `self.epoch`, và `train.py:160` gọi `train_ds.set_epoch(epoch)` mỗi epoch
  → **dữ liệu train thật sự đổi mỗi epoch** (blur, exposure, JPEG quality, shot noise… bốc lại).
  Đây là một nguồn dao động loss hợp lệ, không phải bug.
- `val_ds` **không** được gọi `set_epoch` → luôn dùng seed `epoch=0`
  → nhiễu trên tập val cố định, các epoch so sánh được với nhau. ✅

---

## 4. Giải thích từng chỉ số

### `z_std` — đồng hồ báo collapse (quan trọng nhất)

**Công thức** — `train/losses.py:52-59`:

```python
z_tgt.reshape(-1, D).std(dim=0).mean()
```

Lấy độ lệch chuẩn theo trục `(batch, token)` cho **từng chiều** trong 384 chiều, rồi trung bình.

**Tính trên:** `z_tgt` = đầu ra `target_encoder` chạy trên **ảnh/IMU SẠCH**, có `stop-grad`
(`models/jepa.py:92-97`). Chỉ có ở log train.

**Vì sao quanh 1.0:** `context_encoder` và `target_encoder` kết thúc bằng `nn.LayerNorm`
(`models/encoder.py:48,53`), nên mỗi vector token có scale ~1 theo thiết kế.
LayerNorm chuẩn hoá theo *chiều đặc trưng của từng token*, còn `z_std` đo *phương sai giữa các
token/mẫu* — nên nó vẫn phát hiện được collapse.

| Giá trị | Nghĩa |
|---|---|
| 0.7 – 1.0, phẳng | ✅ encoder tạo biểu diễn khác nhau cho đầu vào khác nhau |
| tụt đều 1.0 → 0.5 → 0.2 | 🔴 **representation collapse** |
| < 0.2 | 🔴 encoder trả gần như cùng một vector cho mọi ảnh |

**Đây là cái bẫy lớn nhất của JEPA.** Model có thể làm `L_jepa` → 0 bằng cách cho encoder
xuất hằng số. Loss đẹp, model vô dụng. Luôn đọc `L_jepa` **cùng với** `z_std`.

---

### `L_jepa` — mục tiêu chính

**Công thức** — `train/losses.py:21-22`:

```python
F.smooth_l1_loss(z_pred, z_tgt.detach())
```

**Tính trên:** chỉ các token thuộc `target_index` (~224/512 token). `z_pred` do predictor
sinh ra từ context **NHIỄU**; `z_tgt` từ target encoder chạy trên dữ liệu **SẠCH**.
Có ở cả log train (1 batch) và log eval (trung bình 100 batch).

**Thang quy đổi.** Vì `z` đã qua LayerNorm nên coi như phân phối chuẩn đơn vị.
Số dưới đây tính bằng Monte-Carlo 4 triệu mẫu:

| `L_jepa` | Sai số dư (đơn vị std) | Đánh giá |
|---|---|---|
| **0.4245** | 1.00 | = dự đoán bằng 0. Không học được gì |
| 0.2927 | 0.80 | mới bắt đầu học |
| 0.1746 | 0.60 | khá |
| 0.1235 | 0.50 | khá tốt |
| 0.0798 | 0.40 | tốt |
| 0.0450 | 0.30 | rất tốt |
| 0.0200 | 0.20 | 🔴 kiểm tra `z_std` ngay |
| 0.0050 | 0.10 | 🔴 gần như chắc chắn collapse |

**Lưu ý về "mục tiêu di động":** target encoder cập nhật bằng EMA với
`m ≈ 0.9966` ở giai đoạn đầu, nên `z_tgt` vẫn đang dịch chuyển. `L_jepa` giai đoạn đầu
không phải một hàm cố định để hội tụ — nó đang đuổi theo một cái đích đang chạy.
Khi `m → 1.0` ở cuối lịch, đích đứng yên và `L_jepa` mới ổn định thật.

---

### `L_img` — tái tạo ảnh

**Công thức** — `train/losses.py:17-18`:

```python
torch.sqrt((x - y)**2 + 1e-6).mean()      # Charbonnier ≈ MAE mượt
```

**Tính trên:** `img_rec` so với `batch["img_clean"]`, toàn bộ ảnh 256×256, thang `[0, 1]`.
Chỉ có ở log train.

| `L_img` | Nghĩa |
|---|---|
| 0.193 | = trả về màu trung bình của ảnh. Vô dụng |
| **0.076** | = trả về đúng ảnh nhiễu đầu vào — **ngưỡng phải vượt** |
| < 0.06 | tốt (sai số ~15/255 mỗi pixel) |
| < 0.04 | rất tốt (~10/255) |

---

### `L_imu` — tái tạo IMU

**Công thức** — `train/losses.py:25-28`:

```python
F.l1_loss(rec[..., 0:3], clean[..., 0:3]) + F.l1_loss(rec[..., 3:6], clean[..., 3:6])
```

Là **tổng hai số** (acc + gyro), nên thang gấp đôi một L1 đơn.

**Đơn vị:** IMU đã chuẩn hoá z-score bằng `configs/norm_stats.yaml`
(`data/normalize.py:29-34`), nên đơn vị là **số lần độ lệch chuẩn**, không phải m/s².

| `L_imu` | Nghĩa |
|---|---|
| ≈ 1.6 | = trả về 0 (tức là mean). Vô dụng |
| **0.177** | = trả về đúng IMU nhiễu đầu vào — **ngưỡng phải vượt** |
| < 0.15 | tốt |
| < 0.10 | rất tốt |

---

### `L` — tổng

**Công thức** — `train/losses.py:40`:

```python
L = L_jepa + lambda_img * L_img + lambda_imu * L_imu     # cả hai lambda = 1.0
```

Chỉ là tổng để backward. Nó bị chi phối bởi số hạng nào đang lớn nhất, nên
**không dùng để đánh giá chất lượng**. Luôn nhìn 3 thành phần tách riêng.

---

### `PSNR` — chỉ có ở dòng eval

**Công thức** — `train/engine.py:116-117`:

```python
mse  = ((img_rec.float() - img_clean.float())**2).mean()
psnr = -10 * log10(mse)
```

MAX = 1.0 vì ảnh ở thang `[0, 1]`. **Tính trên tập valid**, trung bình có trọng số theo batch.

| PSNR | Nghĩa |
|---|---|
| ~3 dB | lúc mới khởi tạo. Recon head là `nn.Linear` không có sigmoid (`models/recon_head.py:43`) nên đầu ra không bị chặn trong `[0,1]` → MSE ban đầu rất lớn. **Bình thường.** |
| **14.0 dB** | đoán màu trung bình của ảnh — sàn tuyệt đối |
| **20.9 dB** | ⭐ PSNR của **chính ảnh nhiễu đầu vào** (p10 = 16.2, p90 = 24.7) |
| 18 – 22 dB | tốt (xem mục 5 để hiểu vì sao) |
| > 24 dB | rất tốt |

---

### `RMSE acc` / `RMSE gyro` — chỉ có ở dòng eval

**Công thức** — `train/engine.py:118-124`: RMSE riêng cho 3 kênh acc và 3 kênh gyro,
so `imu_rec` với `imu_clean`, **trên tập valid**.

**Đơn vị: số lần std.** Quy ra vật lý bằng `configs/norm_stats.yaml`:

```
acc_std  = [8.11, 7.97, 5.48] m/s²   (trung bình ≈ 7.2)
gyro_std = [0.61, 0.68, 0.63] rad/s  (trung bình ≈ 0.64)
```

| RMSE (std) | Nghĩa | acc (m/s²) | gyro (rad/s) |
|---|---|---|---|
| 1.00 | = đoán mean. Vô dụng | 7.2 | 0.64 |
| **0.117 / 0.121** | ⭐ mức nhiễu của tín hiệu đầu vào — **phải vượt** | 0.84 | 0.077 |
| 0.08 | tốt | 0.58 | 0.051 |
| 0.05 | rất tốt | 0.36 | 0.032 |

---

### `lr`, `m`, `im/s` — chỉ số vận hành, không phải chất lượng

**`lr`** — `train/train.py:60-69`: warmup tuyến tính 500 bước rồi cosine về 0.
`total_steps = epochs × limit_train_batches = 40 × 500 = 20 000`
(**không** phụ thuộc `batch_size`).

| epoch | lr | % đỉnh |
|---|---|---|
| 1 | 1.50e-4 | 100% |
| 5 | 1.44e-4 | 96% |
| 10 | 1.27e-4 | 85% |
| 20 | 7.2e-5 | 48% |
| 30 | 1.9e-5 | 13% |
| 39 | ~0 | 0% |

→ Loss "phẳng" ở epoch 5–15 thường chỉ là **noise floor của lr 1.5e-4**, chưa phải hội tụ.
Đợt giảm thứ hai thường rơi vào epoch 20–35 khi lr tụt.

**`m`** — `train/__init__.py:4-9`: momentum EMA đi từ 0.996 → 1.0 theo cosine.
Ở step 2 750 mới `m ≈ 0.9966`. Dùng để kiểm tra lịch chạy đúng, nhất là sau `--resume`.

> Sau khi resume phải giữ nguyên `epochs` và `limit_train_batches`, nếu không `total_steps`
> đổi → lịch `lr` và `m` lệch.

**`im/s`** — throughput. Tụt = dataloader nghẽn (Kaggle chỉ có 4 vCPU). Không liên quan chất lượng.

---

## 5. Vì sao PSNR ở đây thấp hơn denoiser thông thường

Recon head **không nhìn thấy cả ảnh**. `models/jepa.py:66-69` ghép `emb_full` từ ba nguồn.
Đo thật trên 200 lần bốc mask (`train/masking.py:90-135`):

```
Tổng N = 512 token  (256 ảnh + 256 IMU)

  context giữ lại   144 token (28%)  ← dữ liệu THẬT, nhưng là bản NHIỄU
  target            224 token (44%)  ← z_pred, do predictor đoán
  bị vứt            144 token (28%)  ← thay bằng missing_token (một hằng số học được)
```

28% bị vứt là do `masking.context_keep_ratio: 0.5` — chỉ giữ một nửa số token ngoài vùng target.

→ Model phải **vừa khử nhiễu vừa vá phần thông tin bị xoá trắng**, trong khi baseline 20.9 dB
là "ảnh nhiễu nhưng đủ 100% pixel". Vì vậy **18–22 dB ở đây đã là kết quả mạnh**;
đừng lấy chuẩn 30 dB của denoiser thuần ra so sánh.

> **Đã sửa:** trước đây dải thô `L3 LL` (4 token mang toàn bộ độ sáng/bố cục) cũng bị vứt,
> 67% số lần mất ít nhất một token, kéo trần PSNR xuống **17.5 dB — thấp hơn cả ảnh nhiễu
> đầu vào**. Nay `masking.protect_coarse: true` luôn giữ dải `LL` (ảnh) và `A` (IMU) trong
> context, trần lên **24.5 dB**.

---

## 6. Ba cảnh báo khi đọc log

**1. `L_jepa` TĂNG trong vài epoch đầu là bình thường — và là dấu hiệu TỐT.**
Khi `z_ctx` bung ra từ ~0.3 lên ~1.0, `z_tgt` bám theo qua EMA và bài toán dự đoán trở nên
khó thật sự. `L_jepa` đi lên trong lúc `z_std` đi lên nghĩa là model đang giải bài toán thật
thay vì đi đường tắt. Chỉ lo khi `z_std` tụt.

**2. Loss train là một batch.** Xem mục 2.1. Muốn so được với dòng `[eval eN]` thì phải cộng dồn
`logs` trong `train_one_epoch` và in một dòng tổng kết cuối epoch.

**3. `L_imu` đang chiếm phần lớn gradient.** Với `lambda_imu: 1.0`, `L_imu` (~1.0–1.6) lớn hơn
`L_jepa` và `L_img` cộng lại. Nếu `L_imu` không giảm, hạ `lambda_imu` xuống 0.2 để `L_jepa`
không bị nuốt.

**Hai cảnh báo cũ đã được sửa trong code** (xem mục 9): `early_stop` không còn dùng `L_jepa`,
và mask lúc eval đã cố định seed.

---

## 7. Checklist "model đang tốt"

1. ✅ `z_std` giữ nguyên 0.7 – 1.0 suốt quá trình train — **quan trọng nhất**
2. ✅ `PSNR` val tăng đều, bò lên vùng 18 dB+ và tiến về 20.9 dB
3. ✅ `RMSE acc` < 0.117 và `RMSE gyro` < 0.121
4. ✅ `L_jepa` giảm **đồng thời** `z_std` không giảm
5. ✅ `L_jepa` val không lệch quá xa `L_jepa` train (chưa overfit)
6. ✅ Đừng kết luận trước epoch 20 — lịch cosine chưa hạ lr

---

## 8. Phụ lục: số liệu baseline gốc

Lệnh:

```bash
python -m qwt_jepa.scripts.baseline_metrics --split valid --n 128
```

Kết quả (`n = 128` mẫu lấy thưa đều trên 5 118 mẫu valid, seed corruption `epoch = 0`):

```
chỉ số                                 mean      p10      p90
psnr_noisy                           24.471   19.715   30.152     ← PSNR của ảnh NHOÈ đầu vào
psnr_mean                            14.033   10.033   17.195     ← đoán màu trung bình
limg_noisy                            0.042    0.018    0.072     ← L_img của ảnh nhoè đầu vào
limg_mean                             0.193    0.094    0.282     ← L_img khi đoán màu trung bình
rmse_acc                              0.115    0.055    0.175     ← nhiễu acc đầu vào
rmse_gyro                             0.117    0.060    0.188     ← nhiễu gyro đầu vào
limu_noisy                            0.171    0.088    0.259     ← L_imu của IMU nhiễu đầu vào
```

Và **độ nét** — chỉ số mà PSNR không nói lên được:

```
ảnh SẠCH             0.0530
ảnh NHOÈ đầu vào     0.0400     ← mờ hơn sạch 25%
```

Mục tiêu "model nét hơn đầu vào" giờ **có nghĩa**. Với config nhiễu cũ, đầu vào có độ
nét 0.0678 > ảnh sạch 0.0587 (nét giả do nhiễu tần số cao) nên mục tiêu đó bất khả thi
về mặt toán học — bộ lọc tuyến tính tối ưu buộc phải làm mượt, nét hơn đầu vào −27%.
Với config nhoè: bộ lọc tối ưu cho **+5%** nét hơn đầu vào, đạt 73% độ nét ảnh sạch.

Khoảng p10–p90 rộng vì `corruption.image.condition` bốc ngẫu nhiên "điều kiện chụp" cho từng
mẫu — có mẫu gần như sạch, có mẫu rất tối và nhoè. Nên PSNR của **một** batch dao động mạnh
là chuyện bình thường; chỉ so sánh giá trị đã trung bình trên 100 batch val.

---

## 9. Các fix chống collapse (đã áp dụng)

Bản train đầu tiên collapse ngay trong epoch 0: `z_std` 0.289 → 0.13, `L_jepa` 0.561 → 0.011,
PSNR đứng ở 10 dB (thấp hơn baseline đoán màu trung bình 14.0 dB). Dưới đây là nguyên nhân
và cách sửa.

### 9.1 Tokenizer dùng chung phá vỡ cách ly EMA — `models/jepa.py`

Trước đây chỉ `context_encoder` có bản EMA; `self.tokenizer` dùng chung cho cả hai nhánh.
`torch.no_grad()` ở nhánh target chặn gradient chảy *qua* nó, nhưng **không** ngăn optimizer
cập nhật tham số tokenizer từ nhánh context. Mỗi bước, `z_tgt` dịch chuyển theo đúng hướng
optimizer muốn → tồn tại đường tắt: thu nhỏ phần phụ thuộc dữ liệu trong token cho đến khi
mọi token giống nhau thì `L_jepa` → 0.

Nay nhánh target là bản EMA **hoàn chỉnh** (`target_tokenizer` + `target_encoder`), và
`ema_update` cập nhật cả hai cặp.

### 9.2 Luôn giữ dải thô trong context — `train/masking.py`

`coarse_tokens()` gom token dải `LL` (ảnh) và `A` (IMU) — 36/512 token. Chúng không bao giờ
làm target và không bao giờ bị vứt. Bật/tắt bằng `masking.protect_coarse`.

### 9.3 Ràng buộc phương sai — `train/losses.py`

```python
variance_loss(z) = relu(gamma - std_theo_chieu(z)).mean()
```

Chỉ áp cho `z_ctx`. **Không** áp cho `z_pred`: predictor kết thúc bằng `Linear` không chuẩn hoá
nên model sẽ "lách" bằng cách phóng to scale đầu ra thay vì tăng đa dạng thật.
Chỉnh bằng `train.loss.lambda_var` và `var_gamma`.

### 9.4 Metric early stop — `configs/base.yaml`

`L_jepa` → `psnr` (`mode: max`, `min_delta: 0.05` dB). Dùng `L_jepa` thì `best.pt` sẽ lưu đúng
checkpoint collapse.

### 9.5 Mask cố định seed khi eval — `train/engine.py`

`evaluate()` tạo `torch.Generator` seed `train.eval_mask_seed` (mặc định 1234), nên mọi epoch
đo trên cùng một chuỗi mask.

### 9.6 Bằng chứng A/B

150 bước, batch 8, cùng seed, dữ liệu thật:

| | `z_std` đầu | `z_std` cuối | thay đổi | `L_jepa` cuối |
|---|---|---|---|---|
| Code cũ | 0.304 | 0.172 | **−43%** | 0.0120 (giả) |
| Code mới | 0.225 | 0.664 | **+195%** | 0.2780 (thật) |

Chạy thật `train.py` 4 epoch ngắn sau khi sửa:

```
[eval e0] L_jepa 0.3414 | z_std 0.614 | PSNR 4.38 dB | RMSE acc 0.4072 gyro 0.2800
[eval e1] L_jepa 0.2204 | z_std 0.904 | PSNR 5.62 dB | RMSE acc 0.2934 gyro 0.1693
[eval e2] L_jepa 0.1247 | z_std 0.962 | PSNR 7.20 dB | RMSE acc 0.2596 gyro 0.1244
[eval e3] L_jepa 0.0695 | z_std 0.973 | PSNR 8.60 dB | RMSE acc 0.2685 gyro 0.1417
```

`z_std` leo lên 0.97 và đứng vững, `L_jepa` giảm, PSNR tăng đều — đúng dạng học thật.

### 9.7 Test hồi quy

`qwt_jepa/tests/test_anti_collapse.py` chốt lại:

- `target_tokenizer` là bản riêng, không nhận gradient, không đổi sau `optimizer.step()`,
  và **có** đổi sau `ema_update()`
- dải thô luôn nằm trong context, không bao giờ là target
- `variance_loss` phạt đúng chiều

### 9.8 Việc còn lại (chưa làm)

**Chuẩn hoá hệ số QWT theo dải.** Biên độ hệ số chênh 56× giữa các dải
(ảnh `L1 HH` std 0.040 vs `L3 LL` std 2.23) nhưng `Tokenizer.image_proj` và `ImageHead.proj`
chỉ là **một** `nn.Linear` dùng chung. Nên chia hệ số mỗi dải cho std cố định của nó rồi nhân
lại trong head. Để riêng vì đụng cả tokenizer lẫn head — làm sau khi xác nhận `z_std` đứng vững.


---

## 10. Collapse kiểu VỊ TRÍ (phát hiện sau mục 9)

Sau khi vá collapse ở mục 9, lần train tiếp theo cho `z_std 1.006`, `L_var 0.008` — trông
hoàn hảo — nhưng `L_jepa` vẫn tụt về **0.0125**, đúng bằng con số của lần collapse trước.

Mổ `z_tgt` ra thì thấy:

```
z_std đang log (gộp cả batch và token) : 0.9949   ← trông rất khoẻ
std theo NỘI DUNG (đổi ảnh)            : 0.1313   ← thực chất chỉ có bấy nhiêu
std theo VỊ TRÍ   (đổi token)          : 0.9946
```

Phép thử quyết định — dự đoán `z_tgt` **chỉ bằng trung bình theo vị trí token**, bỏ qua
hoàn toàn ảnh đầu vào:

```
chỉ dùng vị trí, không nhìn ảnh : 0.0092
model thật (có nhìn ảnh)        : 0.0134     ← TỆ HƠN
```

Model đang thua chính cái baseline không thèm nhìn ảnh.

**Nguyên nhân.** `variance_loss` ở mục 9.3 tính std gộp cả `(batch, token)`. Encoder thoả
mãn ràng buộc đó bằng cách cho các **vị trí** khác nhau — không cần các **ảnh** khác nhau.
Vị trí token là thứ dễ đoán nhất, nên tối ưu hoá `L_jepa` đẩy thẳng vào lối đó.

**Cách sửa.** `variance_loss` và `content_std` nay đo std **theo batch tại từng vị trí token**:

```python
std = torch.sqrt(z.float().var(dim=0) + eps)   # [T, D] - đổi ảnh thì đổi bao nhiêu
return F.relu(gamma - std).mean()
```

Thêm `jepa_loss_position_only()` in ra ở dòng eval làm mốc gian lận thường trực.

**Chọn `gamma`.** Đo A/B 500 bước:

| `gamma` | `L_jepa` | pos-only | tỉ lệ | `z_tgt` nội dung |
|---|---|---|---|---|
| **1.0** | 0.0409 | 0.2067 | **0.20** | **0.662** |
| 0.5 | 0.0166 | 0.0489 | 0.34 | 0.314 |

`gamma 1.0` thắng ở cả hai mặt. Đừng hạ xuống.

**Kết quả sau khi sửa** (`train.py`, 4 epoch × 120 batch, batch 32):

```
[eval e0] L_jepa 0.2152/0.0128pos | z_std 0.361 | PSNR  4.90 dB
[eval e1] L_jepa 0.1159/0.0318pos | z_std 0.488 | PSNR  7.88 dB
[eval e2] L_jepa 0.0812/0.0536pos | z_std 0.520 | PSNR  9.98 dB
[eval e3] L_jepa 0.0510/0.0569pos | z_std 0.511 | PSNR 10.12 dB
```

Tỉ lệ đi 16.8 → 3.6 → 1.5 → **0.90**, vượt mốc 1.0.

**Bài học chung:** một chỉ số chống collapse chỉ chặn được đúng kiểu collapse mà nó đo.
`z_std` gộp chặn được "mọi token về một vector" nhưng mù trước "biểu diễn chỉ còn là hàm
của vị trí". Mốc `pos-only` khó lách hơn vì nó hỏi thẳng: *bỏ ảnh đi thì có tệ hơn không?*
