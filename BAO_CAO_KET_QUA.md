# Báo cáo kết quả dehazing

## 1. Phạm vi và cấu hình

Các mô hình được chạy trên cùng tập test gồm SOTS-Indoor, SOTS-Outdoor, O-HAZE và I-HAZE. Kết quả được tính bằng PSNR, SSIM và thời gian trung bình trên mỗi ảnh. UDPNet sử dụng pipeline Depth Anything V2 + UDPNet; MB-TaylorFormerV2 sử dụng checkpoint OTS-B.

Trong lần chạy này, để giới hạn bộ nhớ GPU, ảnh được giới hạn ở `max-side=512` và chạy batch nhỏ. Vì vậy đây là kết quả đánh giá thực tế trên cấu hình Colab hiện tại, không phải phép tái lập nguyên bản toàn bộ điều kiện huấn luyện/đánh giá của paper.

## 2. Kết quả theo dataset

| Mô hình | SOTS-Indoor PSNR / SSIM | SOTS-Outdoor PSNR / SSIM | O-HAZE PSNR / SSIM | I-HAZE PSNR / SSIM |
|---|---:|---:|---:|---:|
| UDPNet | 21.39 / 0.9157 | **35.52 / 0.9879** | **18.27 / 0.7540** | 16.66 / 0.7626 |
| MB-TaylorFormerV2 | 21.17 / 0.9063 | 35.45 / **0.9883** | 18.23 / 0.7552 | **15.97 / 0.7650** |
| GridDehazeNet | **33.21 / 0.9860** | 31.47 / 0.9839 | 17.32 / 0.6625 | 14.40 / 0.7232 |
| CAP | 22.76 / 0.8833 | 21.44 / 0.9211 | 16.78 / 0.6511 | 15.12 / 0.7348 |
| DCP | 21.70 / 0.8915 | 17.10 / 0.8623 | 14.30 / 0.5722 | 11.54 / 0.5886 |

UDPNet đạt kết quả tốt nhất giữa hai mô hình mới trên SOTS-Indoor, SOTS-Outdoor và O-HAZE. MB-TaylorFormerV2 có SSIM nhỉnh hơn UDPNet trên SOTS-Outdoor và I-HAZE, nhưng PSNR thấp hơn trên các tập này.

## 3. Kết quả theo miền dữ liệu và tốc độ

| Mô hình | Real PSNR / SSIM | Synthetic PSNR / SSIM | RAM peak (GB) | FPS toàn pipeline | FPS suy luận | VRAM peak (GB / 80 GB) |
|---|---:|---:|---:|---:|---:|---:|
| UDPNet | 17.60 / 0.7576 | **28.46 / 0.9518** | 1.846 | **4.3373** | **7.2815** | 23.323 |
| MB-TaylorFormerV2 | **17.29 / 0.7593** | 28.31 / 0.9473 | 1.736 | 3.3097 | 3.3097 | 30.816 |
| GridDehazeNet | 16.10 / 0.6878 | **32.34 / 0.9850** | 1.790 | 2.3004 | 7.6970 | 19.290 |
| CAP | 16.08 / 0.6860 | 22.10 / 0.9022 | 1.190 | 1.8623 | 4.1056 | - |
| DCP | 13.15 / 0.5790 | 19.40 / 0.8769 | 1.290 | 2.3988 | 7.8008 | - |

UDPNet có tốc độ tổng thể cao hơn MB-TaylorFormerV2 trong cấu hình đã chạy. FPS suy luận của UDPNet không bao gồm toàn bộ chi phí chuẩn bị dữ liệu như đọc ảnh và một số bước pipeline; FPS toàn pipeline phù hợp hơn khi ước lượng tốc độ sử dụng thực tế.

## 4. Diễn giải giới hạn kết quả

Không nên so sánh trực tiếp các con số trên với số cao nhất trong paper như một phép tái lập tuyệt đối, vì:

- ảnh đã được thu nhỏ về `max-side=512` để tránh OOM; việc này làm mất chi tiết và thường làm giảm PSNR/SSIM;
- batch size và cách padding khác với pipeline gốc của từng repo;
- UDPNet có thêm Depth Anything V2, nhưng phiên bản backbone, preprocessing và depth prior có thể khác cấu hình tác giả dùng để báo cáo;
- checkpoint được chạy là checkpoint tải được, còn paper có thể dùng checkpoint/phiên bản code, dataset split và hậu xử lý khác;
- paper thường báo cáo với quy trình benchmark cố định, trong khi kết quả này dùng một evaluator chung cho nhiều phương pháp;
- giới hạn VRAM khiến không thể chạy nguyên ảnh độ phân giải cao theo đúng protocol mà không chia tile. Tiled inference giữ chi tiết tốt hơn nhưng chậm hơn nhiều, nên lần chạy cuối ưu tiên tốc độ bằng resize.

Do đó, các kết quả hiện tại nên được hiểu là benchmark so sánh trong cùng môi trường, không phải khẳng định mô hình kém hơn paper.

## 5. Giải thích miền tần số trong paper HazeWaveNet

`paper/Dehaze wavelet.pdf` mô tả **HazeWaveNet**, không phải UDPNet hay MB-TaylorFormerV2. Paper dùng biến đổi wavelet rời rạc (DWT), cụ thể là wavelet Haar, để tách đặc trưng theo tần số và độ phân giải.

### 5.1. DWT tách ảnh như thế nào?

Ở mỗi mức phân rã, DWT áp dụng bộ lọc thông thấp và thông cao theo hai chiều rồi giảm một nửa kích thước không gian. Kết quả gồm bốn dải:

- `LL`: thành phần xấp xỉ/thấp tần, chứa vùng trơn, cấu trúc lớn, độ sáng và tương phản tổng thể;
- `LH`: chi tiết theo một hướng;
- `HL`: chi tiết theo hướng còn lại;
- `HH`: chi tiết biên/texture mạnh, thường được gọi chung là các dải cao tần.

Với Haar, phép tính gần như lấy trung bình và sai phân của các pixel lân cận. `LL` giữ thông tin mượt và giảm kích thước; `LH/HL/HH` giữ biến thiên nhanh như cạnh, texture và biên vật thể. Paper dùng DWT nhiều mức, thường là bốn mức, nên `LL` tiếp tục được phân rã để tạo biểu diễn đa tỉ lệ.

### 5.2. Mạng làm gì với các dải tần?

1. **Nhánh thấp tần:** xử lý `LL` bằng các nhóm FEGB. Đây là nhánh chính để sửa veil haze, tương phản toàn cục và vùng mờ. Guidance từ Dark Channel Prior được chuyển thành bản đồ truyền qua/trend map rồi dùng để điều chế attention và convolution theo vùng có haze mạnh.
2. **Nhánh cao tần:** xử lý `LH/HL/HH` bằng các module nhẹ VP/VR. Mục tiêu không phải khử haze mạnh mà bù suy giảm tương phản cạnh, giữ biên và texture, tránh ảnh bị nhòe.
3. **Wavelet Integration Module (WIM):** hợp nhất các nhánh thấp/cao tần ở nhiều mức bằng convolution, nối đặc trưng, residual connection và upsampling học được. Paper không dùng IDWT trực tiếp ở bước cuối; WIM học cách tái tạo để giảm ringing/checkerboard artifact.
4. **Refinement:** convolution cuối và Tanh tạo ảnh dehazed.

### 5.3. Hàm mất mát miền tần số

Paper dùng tổng hợp hai thành phần:

```text
L = theta * La + (1 - theta) * Ld
```

Trong đó `La` so sánh biên độ biến đổi của ảnh dự đoán và ảnh sạch; `Ld` là sai số trên từng dải `LL/LH/HL/HH`. Vì vậy mạng không chỉ tối ưu ảnh ở miền pixel mà còn bị ràng buộc phải khôi phục đúng cấu trúc đa tần số.

### 5.4. Ý nghĩa trực giác

Haze chủ yếu làm giảm tương phản và che phủ cấu trúc lớn nên ảnh hưởng mạnh đến `LL`, nhưng không phải hoàn toàn không ảnh hưởng cao tần: biên và tương phản cục bộ cũng bị suy giảm. Vì vậy HazeWaveNet sửa mạnh nhánh thấp tần, đồng thời bảo toàn và tinh chỉnh nhẹ các dải cao tần thay vì bỏ qua chúng.

## 6. Kết luận

Trong cấu hình thực nghiệm hiện tại, UDPNet là lựa chọn cân bằng tốt hơn giữa chất lượng và tốc độ; MB-TaylorFormerV2 cho SSIM tốt ở một số tập nhưng chậm hơn và nhạy với giới hạn batch/VRAM. Kết quả thấp hơn paper là có thể giải thích được bởi resize `512`, preprocessing và điều kiện chạy khác. Nếu cần tái lập sát paper, bước tiếp theo là chạy full-resolution hoặc tiled inference, dùng đúng split/preprocessing/checkpoint của từng công trình và chấp nhận thời gian infer cao hơn.
