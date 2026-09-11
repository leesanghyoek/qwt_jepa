# Model này là gì, và mỗi mảnh trong nó để làm gì

Tài liệu này giải thích **ý nghĩa** của kiến trúc — vì sao từng mảnh tồn tại, bằng chứng
đo đạc nào dẫn đến nó, và nó làm được / không làm được gì.

Sơ đồ luồng dữ liệu chi tiết: `KIEN_TRUC_MODEL.md`.
Cách đọc log và các ngưỡng: `LOG_TRAIN_GIAI_THICH.md`.
Quy trình chạy trên Kaggle: `kaggle_session.txt`.

---

## 1. Bài toán

Đầu vào là **ảnh mờ + IMU nhiễu** từ drone. Muốn ra **ảnh nét hơn + IMU mượt, hết spike**.

Điểm đặc biệt: hai tín hiệu này **không độc lập**. Nhoè chuyển động trong một khung hình
chính là tích phân chuyển động camera trong thời gian phơi sáng — mà gyro đo đúng thứ đó
ở 100 Hz. Nên IMU **chứa thông tin về cách khử nhoè ảnh**, và ảnh giúp kiểm chứng IMU.

Đó là lý do chính đáng để xử lý chung, và là lý do dùng QWT: cả hai đều được đưa về
**miền tần số đa thang**, nơi nhiễu và nhoè biểu hiện rõ ràng theo từng dải.

---

## 2. Model gồm ba phần

```
              QWT  ──►  tokenizer  ──►  512 token (256 ảnh + 256 IMU)
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    │                                                   │
              LƯỢT CHE (học)                                   LƯỢT ĐỦ (làm việc)
              152 token                                        cả 512 token
                    │                                                   │
              context_encoder ◄═══ CÙNG MỘT MODULE ═══► context_encoder
                    │                                                   │
               predictor                                    image_head / imu_head
                    │                                                   │
                 L_jepa                                       ảnh sạch, IMU sạch
```

| Phần | Tham số | Vai trò |
|---|---:|---|
| `tokenizer` + `context_encoder` | 21.80M | **xương sống** — biến tín hiệu thành biểu diễn |
| `image_head` + `imu_head` | 1.77M | **bộ giải mã** — biểu diễn → ảnh/IMU sạch |
| `predictor` + nhánh target EMA | 23.82M | **chỉ dùng lúc học**, bỏ hết khi triển khai |

Tổng 47.39M, học 25.60M, triển khai chỉ cần **23.57M**.

---

## 3. Bảy quyết định thiết kế, và bằng chứng cho mỗi cái

### 3.1 Chuẩn hoá hệ số theo dải — `band_norm`

**Vấn đề đo được:** biên độ hệ số QWT chênh nhau **123 lần** giữa các dải.

```
L3 LL   4 token (1.6%)   giữ 94% năng lượng   std 2.60
L1 HH  64 token (25%)    giữ 0.08%            std 0.021
```

Mà tokenizer chỉ có **một** `nn.Linear` dùng chung. Hệ số dải mịn đi qua nó cho activation
nhỏ xíu → encoder gần như **không nhìn thấy chi tiết mịn** → ảnh ra mờ.

**Sửa:** chia mỗi dải cho std riêng, nhân lại ở head. Activation vào encoder từ chênh
107× còn **2×**.

### 3.2 Tái tạo từ TOÀN BỘ token — `recon_from_full`

**Vấn đề:** recon head trước đây đọc `emb_full` = 30% token thật + 41% token predictor
đoán + 30% hằng số. Mà đo được predictor còn thua mẹo "đoán theo vị trí" (tỉ số 1.1) nên
41% kia là rác. Head chỉ có **30% thông tin thật** để dựng cả bức ảnh.

**Sửa:** một lượt chạy riêng qua **cùng** `context_encoder` trên cả 512 token.
`RMSE acc` giảm một nửa ngay lập tức. Cũng sửa luôn lệch train/test: trước đây encoder
lúc train chưa bao giờ thấy đủ 512 token nhưng lúc triển khai lại nạp cả 512.

### 3.3 Skip connection hệ số thô — `recon_skip`

Chi tiết mịn phải sống sót qua 12 lớp attention toàn cục rồi được bung ra bởi **một**
lớp Linear. Nối thẳng hệ số QWT thô vào đầu vào head.

**Phải là hệ số THÔ, không phải đầu ra tokenizer** — tokenizer là `Linear(768 → 384)`,
đã nén mất một nửa thông tin.

### 3.4 Head có cổng — `recon_gate` + `recon_gate_max`

**Vấn đề:** khử nhiễu wavelet về bản chất là **co hệ số** — `ŵ = w × g(w, ngữ cảnh)`, một
phép **nhân**. Một lớp Linear chỉ **cộng** được. Và `proj` dùng chung cho cả 10 dải nên
không thể cho `L1 HH` hệ số 0.3 còn `L3 LL` hệ số 1.05.

Trần đo được theo độ phức tạp của head:
```
copy nguyên ảnh vào              19.51 dB
1 hệ số nhân / mỗi dải           20.66 dB   (+1.15)
1 hệ số nhân / mỗi TOKEN         21.82 dB   (+2.31)
```

**Sửa:** `gain = 1 + gate_max·tanh(g)`, `norm = raw × gain + delta`.

Hai tính chất quan trọng:
- **Khởi tạo bằng 0 → đầu ra đúng bằng đầu vào.** Model xuất phát tại "sàn copy" và chỉ
  có thể đi lên, thay vì mò từ 0. Đo lại: lệch 6e-07.
- **`tanh` chặn hệ số trong [0, 2].** Không chặn thì model **phân kỳ**: đo được PSNR tụt
  25 → 12 dB và độ nét lên 2.4 (ảnh ra nét hơn cả ảnh gốc = đang bịa tần số cao).

### 3.5 Nhoè sinh TỪ GYRO — `blur_from_imu`

**Đây là quyết định quan trọng nhất, và nó đến từ việc bạn chỉ ra một khẳng định sai
của tôi.**

Tôi nói "IMU ghi lại chuyển động gây ra nhoè nên model dùng IMU để khử nhoè được".
Kiểm tra code:

```python
def corrupt_image(img, rng, cfg)        # KHÔNG hề nhận imu
angle = float(rng.uniform(0.0, 180.0))  # hướng nhoè bốc ngẫu nhiên
```

Tương quan gyro ↔ nhoè **bằng 0 theo đúng định nghĩa**. Toàn bộ lý do ghép hai modality
không có cơ sở trong dữ liệu.

**Sửa:** `dịch chuyển trên ảnh = ω × T_phơi_sáng × f`. Hiệu chuẩn trên dữ liệu thật
(gyro p50 0.58 rad/s, p90 1.69 rad/s; f = 320 px):

```
exposure_time      nhoè p50   mờ hơn sạch   tương quan gyro↔nhoè
[0.010, 0.025]        2 px        14%            0.865
[0.020, 0.050]        5 px        25%            0.893   ← chọn
[0.040, 0.090]        9 px        36%            0.922
```

Chọn mức giữa vì độ khó **đúng bằng** config ngẫu nhiên cũ, nhưng giờ nhoè **đoán được
từ IMU**.

**Hệ quả đẹp:** nhoè sinh từ gyro **sạch**, model nhìn thấy gyro **nhiễu**. Nên muốn khử
nhoè tốt, model buộc phải khử nhiễu IMU tốt trước. Hai mục tiêu của bạn từ chỗ rời rạc
trở thành **hỗ trợ nhau**.

### 3.6 Config nhiễu nhoè chi phối

Mục tiêu "ảnh nét hơn đầu vào" **chỉ có nghĩa khi đầu vào mờ hơn ảnh sạch**.

```
độ nét ảnh sạch              0.0587
config cũ (nhiễu chi phối)   0.0678   ← NÉT GIẢ, cao hơn cả ảnh sạch
config này (nhoè chi phối)   0.0400   ← mờ hơn sạch 25%
```

Với config cũ, bộ lọc tuyến tính **tối ưu** cho ra ảnh nét hơn đầu vào **−27%** — tức
mục tiêu bất khả thi về mặt toán học. Với config này: **+5%**, đạt 73% độ nét ảnh sạch.

### 3.7 Chống collapse

Ba lớp bảo vệ, mỗi lớp chặn một kiểu collapse đã thực sự xảy ra:

| Cơ chế | Kiểu collapse nó chặn |
|---|---|
| `target_tokenizer` là bản EMA riêng | mọi token về cùng một vector (`z_std` tụt 0.29 → 0.13) |
| `variance_loss` đo theo **nội dung** | biểu diễn chỉ còn là hàm của **vị trí** token |
| `L_jepa_pos` in ở dòng eval | mốc "gian lận" thường trực để đối chiếu |

Bài học: **một chỉ số chống collapse chỉ chặn được đúng kiểu collapse mà nó đo.**

---

## 4. Dùng model như thế nào — HAI GIAI ĐOẠN

Đây là phần quan trọng nhất, và là thứ sai suốt một thời gian dài.

**Không được chạy JEPA và tái tạo cùng lúc.** Hai mục tiêu đối nghịch theo định nghĩa:
JEPA cố tình **vứt** chi tiết vụn (đó là mục đích của kiến trúc), tái tạo cần **giữ** từng
pixel. Quan sát được: `L_jepa` tụt 0.55 → 0.005 trong khi `L_img` **tăng** 0.044 → 0.126,
kèm theo encoder bị giật (`ctx` dao động 0.37–1.08) nên head không bám kịp.

```
GIAI ĐOẠN 1 — tiền huấn luyện
    lambda_img 0 | lambda_imu 0 | lambda_jepa 1 | lambda_var 1
    early_stop: L_jepa (min)
    → encoder học biểu diễn. Không head nào can thiệp.

GIAI ĐOẠN 2 — tinh chỉnh phục hồi
    --init-from <checkpoint GĐ1>  --freeze-backbone 1
    lambda_jepa 0 | lambda_var 0 | lambda_img 1 | lambda_imu 1
    early_stop: psnr (max)
    → chỉ 1.77M/25.60M học. Biểu diễn ĐỨNG YÊN, head có mục tiêu cố định để bám.
```

Kiểm chứng giai đoạn 2:
```
[eval e0] z_std 0.648 | PSNR 23.74 | net 0.769
[eval e1] z_std 0.648 | PSNR 23.69 | net 0.776
[eval e2] z_std 0.648 | PSNR 23.64 | net 0.785
```
`z_std` đứng yên tuyệt đối, `net` bò lên đều mà không vọt quá 1.0. Nhanh hơn ~2.3×.

---

## 5. Thí nghiệm của đề tài

Kiến trúc hai giai đoạn cho phép trả lời đúng câu hỏi khoa học:

> **Tiền huấn luyện QWT-JEPA có giúp phục hồi ảnh + IMU tốt hơn không?**

Chạy giai đoạn 2 **hai lần**, mọi thứ giống nhau trừ điểm xuất phát:

| | |
|---|---|
| (a) | `--init-from runs/gd1/last.pt --freeze-backbone 1` — có tiền huấn luyện |
| (b) | không `--init-from`, `--freeze-backbone 0` — train từ đầu |

> `--init-from` chỉ nạp **trọng số** rồi bắt đầu lại từ epoch 0. Khác `--resume` (dùng để
> tiếp tục một run bị ngắt) — `--resume` khôi phục cả optimizer, scheduler và đặt
> `start_epoch = epoch+1`, nên sau một GĐ1 40 epoch thì GĐ2 sẽ **không train gì cả**.

So `PSNR` / `net` / `RMSE acc` / `RMSE gyro`.

**Nói thẳng:** nhiễu ở đây là **tự sinh** nên có vô hạn cặp (sạch, hỏng) = vô hạn nhãn.
Tự giám sát có lợi nhất khi nhãn **khan hiếm**. Nên kết quả (b) ngang (a) là hoàn toàn
có thể — và đó vẫn là một **kết quả hợp lệ, đáng báo cáo**, không phải thất bại.

---

## 6. Làm được gì, không làm được gì

| Mục tiêu | Trạng thái |
|---|---|
| IMU ít spike/nhiễu | ✅ **đã đạt** — RMSE acc 0.1098 vs mốc 0.123, gyro 0.0964 vs 0.122 |
| Ảnh ít nhiễu | ✅ khả thi |
| Ảnh nét **hơn ảnh vào** | ⚠️ khả thi nhưng khiêm tốn — trần đo được **+5%**, đạt 73% độ nét ảnh sạch |
| Ảnh nét **bằng ảnh gốc** | ❌ cần mục tiêu sinh (GAN/diffusion), đánh đổi PSNR |

**Vì sao không nét bằng ảnh gốc.** Bốn phép đo độc lập cho cùng kết luận: mọi hàm mất mát
hồi quy từng pixel đều có nghiệm tối ưu là ảnh **mờ**.

| Phép thử | Kết quả |
|---|---|
| Làm mờ ảnh nhiễu | loss **giảm** 3.8% |
| Loss cân bằng dải (`lambda_band`) | tối ưu ở sigma 1.5 — mờ **hơn nữa** |
| Số hạng khớp độ nét | không dịch điểm tối ưu |
| Oracle trên dữ liệu nhoè | nét **kém hơn** đầu vào 6% |

Khôi phục tần số đã mất đòi hỏi **bịa ra** chi tiết hợp lý, mà bịa ra luôn **làm tăng**
MSE. Tối ưu MSE và làm nét là hai mục tiêu đối nghịch — đây là đánh đổi
perception–distortion, đã được chứng minh về lý thuyết.

**Và một mốc so sánh sai cần bỏ:** ảnh nhiễu có độ nét **cao hơn** ảnh sạch vì nhiễu
chính là tần số cao. Nên một bộ khử nhiễu làm đúng việc **luôn trông mềm hơn ảnh nhiễu**.
Mốc đúng là so với **ảnh sạch**, không phải ảnh nhiễu.

---

## 7. Đọc log

```
e5 150/300 step 1651 | L 0.40 (jepa 0.19 img 0.052 imu 0.156 band 0.00 var 0.00)
                     | gate 0.08 | z_std 1.13 ctx 1.19 | lr 1.46e-04 m 0.9962 | 50 im/s

[eval e5] L_jepa 0.106/0.458pos | z_std 1.018 | PSNR 19.93 dB | net 0.812 | RMSE acc 0.127 gyro 0.124
```

Bốn cột quan trọng nhất, mỗi cột bắt một kiểu hỏng riêng:

| Cột | Ý nghĩa | Hỏng khi |
|---|---|---|
| `z_std` / `ctx` | độ đa dạng biểu diễn **theo nội dung** | tụt dần → collapse |
| `.../...pos` | `L_jepa` model / `L_jepa` của kẻ chỉ đoán theo vị trí | tỉ số ≥ 1.0 |
| `gate` | biên độ hệ số nhân của head | ~0 = head chưa học |
| `net` | độ nét ảnh ra / độ nét ảnh sạch | > 1.0 = đang bịa tần số cao |

Mốc baseline (chạy lại bằng `scripts/baseline_metrics.py` mỗi khi đổi mức nhiễu):
`psnr_noisy` **25.45 dB**, `rmse_acc` **0.117**, `rmse_gyro` **0.116**, `net` đầu vào ≈ **0.75**.

---

## 8. Điều đáng giá nhất mà quá trình này để lại

Không phải kiến trúc, mà là **bộ công cụ đo**:

- `scripts/baseline_metrics.py` — mốc "copy đầu vào" cho mọi chỉ số
- `L_jepa_pos` — mốc gian lận thường trực cho JEPA
- `net` — bắt được việc bịa tần số cao, thứ PSNR không thấy
- `gate` — cho biết head có đang học không
- 17 test hồi quy chốt lại từng bug đã sửa

Mỗi công cụ ở trên ra đời **sau khi** một lỗi đã âm thầm chạy hàng giờ mà không ai biết.
Ba lần liên tiếp, lỗi chỉ lộ ra khi có chỉ số đo đúng thứ cần đo. Bài học:
**không thêm một cơ chế nào mà không thêm cách nhìn thấy nó đang làm gì.**

---

# Phụ lục: Huấn luyện hai giai đoạn, giải thích từ đầu

Phần này viết cho người chưa quen lý thuyết học máy. Nếu bạn đã nắm rõ, bỏ qua được.

## A. "Huấn luyện" thực chất là gì

Model là **một đống số** — ở đây là 25.60 triệu số thực, gọi là *tham số*. Lúc mới tạo,
chúng là số ngẫu nhiên, và model làm gì cũng sai.

Huấn luyện là vòng lặp bốn bước:

```
1. Cho dữ liệu chạy qua đống số đó       →  ra kết quả
2. Tính LOSS: một con số đo "sai bao nhiêu"
3. Tính GRADIENT: mỗi trong 25.6 triệu số nên tăng hay giảm để loss nhỏ đi
4. Chỉnh từng số một chút theo hướng đó   →  quay lại bước 1
```

Một "bước" (step) là một lần chạy vòng này. Một "epoch" ở đây là 300 bước.

**Điều quan trọng nhất phải nhớ:** model **chỉ học được cái mà loss đo**. Loss không
đo độ nét thì model không học làm nét, dù bạn muốn đến đâu. Toàn bộ tài liệu này xoay
quanh hệ quả của câu đó.

## B. JEPA khác gì model quen thuộc

Model khử nhiễu thông thường (autoencoder) làm thế này:

```
ảnh nhiễu  →  [model]  →  ảnh sạch dự đoán
                              ↓
                    loss = so từng pixel với ảnh sạch thật
```

JEPA làm khác hẳn:

```
ảnh nhiễu  →  [model]  →  MỘT VECTOR 384 số
                              ↓
                    loss = so vector đó với vector "đúng"
```

Đầu ra không phải ảnh, mà là **một dãy số tóm tắt nội dung**. Dãy số đó gọi là
*biểu diễn* (representation) hay *latent*.

**Vì sao lại làm vậy?** Vì trong ảnh có những thứ **không thể đoán được**: hạt nhiễu
cụ thể rơi ở đâu, vân gỗ ngẫu nhiên trông ra sao. Ép model đoán từng pixel là ép nó
tiêu năng lực vào việc bất khả thi. JEPA nói: đừng đoán pixel, hãy đoán **ý nghĩa**.

Cách JEPA tự tạo bài tập cho mình mà không cần ai gán nhãn:

```
1. Cắt ảnh + IMU thành 512 mảnh nhỏ ("token")
2. CHE đi ~70%, chỉ cho model xem ~152 token
3. Bắt model đoán: "phần bị che có latent trông như thế nào?"
4. So với latent thật (tính từ dữ liệu SẠCH đầy đủ)
```

Muốn đoán đúng thì buộc phải *hiểu* cảnh đang nhìn. Đó là toàn bộ ý tưởng.

## C. Giai đoạn 1 — chi tiết từng bước

```
ĐẦU VÀO (mỗi mẫu):
    ảnh nhiễu  [3, 256, 256]      ảnh sạch  [3, 256, 256]
    IMU nhiễu  [128, 6]           IMU sạch  [128, 6]
        └── model được xem ──┘        └── chỉ dùng làm ĐÁP ÁN ──┘

 ┌─ NHÁNH CONTEXT (nhánh chính, đang học) ────────────────────────┐
 │  ảnh+IMU NHIỄU → QWT → 512 token, mỗi token 384 số             │
 │  CHE: chỉ giữ ~152 token                                        │
 │  → context_encoder (12 lớp) → z_ctx  [152, 384]                 │
 │  → predictor → z_pred  [~208, 384]   ← ĐOÁN latent phần bị che  │
 └─────────────────────────────────────────────────────────────────┘

 ┌─ NHÁNH TARGET (làm đáp án, KHÔNG học) ─────────────────────────┐
 │  ảnh+IMU SẠCH → QWT → 512 token (đủ, không che)                 │
 │  → target_encoder → lấy ra ~208 token đúng vị trí bị che         │
 │  → z_tgt  [~208, 384]                                           │
 └─────────────────────────────────────────────────────────────────┘

LOSS:  L_jepa = |z_pred − z_tgt|     ← so HAI DÃY SỐ, không phải hai ảnh
```

**Đầu ra của giai đoạn 1 KHÔNG phải ảnh.** Nó là hai dãy số được đem so với nhau.

Vài chi tiết dễ thắc mắc:

- **Vì sao nhánh target dùng dữ liệu sạch?** Để đáp án là "cảnh thật trông thế nào",
  không phải "cảnh nhiễu trông thế nào". Model học đi từ nhiễu về sạch ngay trong
  không gian latent.
- **`target_encoder` là gì?** Một bản sao của `context_encoder`, **không** học bằng
  gradient. Mỗi bước nó bị kéo nhích về phía bản chính: `target = 0.996×target +
  0.004×context`. Gọi là EMA. Nếu dùng chung một encoder cho cả hai nhánh thì model
  gian lận được — nó chỉ cần làm cả hai bên cùng ra một hằng số là loss về 0 mà không
  học gì. Đó là *collapse*, và nó **đã thực sự xảy ra** trong dự án này.
- **Học những gì:** `tokenizer` + `context_encoder` + `predictor` = **23.82M** tham số.
  Hai recon head có `lambda_img = lambda_imu = 0` nên **không nhận gradient**.

**Kết thúc GĐ1 bạn có:** một `context_encoder` biết biến tín hiệu nhiễu thành biểu diễn
có nghĩa. Nó **chưa biết vẽ lại ảnh** — chưa ai dạy nó việc đó.

## D. Giai đoạn 2 — chi tiết từng bước

```
ĐẦU VÀO:  ảnh nhiễu [3,256,256] + IMU nhiễu [128,6]     ← CHỈ có vậy
                            │
                    QWT → 512 token
                            │
              KHÔNG che gì cả — dùng đủ 512 token
                            │
              context_encoder  ❄️ ĐÓNG BĂNG ❄️
                            │
                   biểu diễn [512, 384]
                     ┌──────┴──────┐
                image_head      imu_head      ← chỉ HAI CÁI NÀY học
                     │              │
ĐẦU RA:      ảnh [3,256,256]   IMU [128,6]    ← tín hiệu THẬT

LOSS:  L_img = |ảnh ra − ảnh sạch|      L_imu = |IMU ra − IMU sạch|
```

**Khác GĐ1 ở ba chỗ:**

| | GĐ1 | GĐ2 |
|---|---|---|
| Che token | có (~152/512) | **không** (đủ 512) |
| Đầu ra | vector latent | **ảnh + IMU** |
| Học | 23.82M | **1.77M** (7% model) |

**Vì sao GĐ2 không che?** Vì che là công cụ để *tạo bài tập* cho GĐ1. GĐ2 không giải
đố nữa — nó làm việc thật: nhận đủ tín hiệu vào, trả tín hiệu sạch ra. Đúng như lúc
triển khai.

## E. "Đóng băng" nghĩa là gì cụ thể

Mỗi tham số có một cờ `requires_grad`. Đóng băng là bật cờ đó thành `False` cho
53 tensor thuộc `tokenizer` + `context_encoder` + `predictor`.

Hệ quả cụ thể:

- Bước 3 (tính gradient) **bỏ qua** chúng
- Bước 4 (chỉnh số) **không đụng** vào chúng
- Chúng giữ nguyên giá trị đã học ở GĐ1, y hệt, suốt GĐ2

Kiểm chứng được trong log — cột `z_std`:

```
[eval e0] z_std 0.648 | PSNR 23.74 | net 0.769
[eval e1] z_std 0.648 | PSNR 23.69 | net 0.776
[eval e2] z_std 0.648 | PSNR 23.64 | net 0.785
     ↑ giống hệt nhau đến chữ số thứ ba — biểu diễn KHÔNG đổi
```

Cột `net` (độ nét) vẫn bò lên đều. Nghĩa là head đang học thật, trên một nền đứng yên.

**Tác dụng phụ:** nhanh hơn ~2.3× vì không phải tính gradient qua 12 lớp encoder.

## F. Vì sao không gộp hai giai đoạn làm một

Đây là thứ đã thử và **thất bại**, có số liệu:

```
L_jepa   0.55 → 0.005     ← JEPA thắng
L_img    0.044 → 0.126    ← tái tạo THUA (loss huấn luyện ĐI LÊN)
ctx      dao động 0.37 ↔ 1.08    ← encoder bị giật liên tục
```

Lý do: **hai mục tiêu đòi hỏi ngược nhau.**

| | JEPA muốn | Tái tạo muốn |
|---|---|---|
| Chi tiết vụn, hạt nhiễu | **vứt đi** — đó là mục đích của nó | **giữ từng pixel** |

Một `context_encoder` không thể vừa vứt vừa giữ. Và vì `L_jepa` (0.2–0.5) lớn hơn
`L_img` (0.04) khoảng 10 lần, nó **thắng** trong cuộc giằng co — encoder bị kéo theo
hướng JEPA, còn head thì phải học một ánh xạ từ nền **liên tục biến dạng**. Không bám kịp.

Tách hai giai đoạn giải quyết đúng chỗ đó: GĐ2 nền đứng yên, head có mục tiêu cố định.

Đây không phải mẹo riêng của dự án này — **MAE, DINO, I-JEPA, SimCLR** đều dùng công
thức "tiền huấn luyện rồi tinh chỉnh". Sai lầm là ta đã chạy song song.

## G. Bảng tra khi chạy

| | Giai đoạn 1 | Giai đoạn 2 |
|---|---|---|
| Config | `lambda_img=0, lambda_imu=0` | `lambda_jepa=0, lambda_var=0` |
| `early_stop` | `L_jepa` (min) | `psnr` (max) |
| Cờ thêm | — | `--init-from .../gd1/last.pt --freeze-backbone 1` |
| `--out` | `runs/gd1` | `runs/gd2a` |
| **Nhìn cột** | `L_jepa/pos` < 1.0, `z_std` 0.7–1.1 | `PSNR`, `net`, `gate`, `RMSE` |
| **Bỏ qua cột** | PSNR, net (head chưa học) | `L_jepa` (đã tắt) |
| Dừng khi | in `dung som` hoặc `xong.` | in `dung som` hoặc `xong.` |

**Đừng dùng `--resume` cho GĐ2.** `--resume` là để nối lại một run bị ngắt: nó khôi
phục cả optimizer, scheduler và đặt `start_epoch = epoch+1`. Sau một GĐ1 chạy 40 epoch,
vòng lặp sẽ là `range(40, 40)` — **rỗng, không train gì cả**. Dùng `--init-from`: chỉ
nạp trọng số rồi bắt đầu lại từ epoch 0.

## H. Một chi phí còn tồn tại

Ở GĐ1, hai recon head **vẫn chạy forward** (chỉ là loss nhân 0 nên không có gradient),
và lượt tái tạo qua đủ 512 token cũng vẫn chạy. Nên GĐ1 **lãng phí khoảng 40% thời
gian** cho phần không dùng đến.

Không ảnh hưởng kết quả, chỉ tốn thời gian. Sửa được bằng một cờ bỏ hẳn nhánh tái tạo
khi `lambda_img = lambda_imu = 0`, nhưng chưa làm.
