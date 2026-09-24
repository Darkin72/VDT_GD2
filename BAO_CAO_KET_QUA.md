# Báo cáo kết quả dehazing

## Phạm vi

Các mô hình được đánh giá trên SOTS-Indoor, SOTS-Outdoor, O-HAZE và I-HAZE. UDPNet và MB-TaylorFormerV2 dùng checkpoint theo miền: `ITS` cho SOTS-Indoor/I-HAZE và `OTS` cho SOTS-Outdoor/O-HAZE. Ảnh được giới hạn `max-side=512` để phù hợp VRAM Colab; kết quả phản ánh cấu hình triển khai hiện tại.

## 1. Sheet Dataset

| Mô hình | SOTS-Indoor PSNR / SSIM / ms | SOTS-Outdoor PSNR / SSIM / ms | O-HAZE PSNR / SSIM / ms | I-HAZE PSNR / SSIM / ms |
|---|---:|---:|---:|---:|
| UDPNet | **37.54 / 0.9919 / 119.77** | 35.26 / 0.9877 / 169.14 | **18.27 / 0.7540 / 159.99** | **11.94 / 0.5247 / 222.04** |
| MB-TaylorFormerV2 | 35.61 / 0.9873 / 151.13 | **35.39 / 0.9879 / 195.56** | 18.23 / 0.7552 / 262.02 | 11.09 / 0.4775 / 323.99 |
| GridDehazeNet | 33.21 / 0.9860 / **42.17** | 31.47 / 0.9839 / **47.52** | 17.32 / 0.6625 / 1126.20 | 14.40 / **0.7232** / 1358.43 |
| CAP | 22.76 / 0.8833 / **19.59** | 21.44 / 0.9211 / **18.75** | 16.78 / 0.6511 / 1264.76 | **15.12 / 0.7348** / 1589.52 |
| DCP | 21.70 / 0.8915 / 47.87 | 17.10 / 0.8623 / 44.19 | 14.30 / 0.5722 / 2335.65 | 11.54 / 0.5886 / 2845.65 |

UDPNet dẫn đầu SOTS-Indoor với `37.54 dB`, cao hơn MB-TaylorFormerV2 `1.93 dB` và GridDehazeNet `4.33 dB`. Việc dùng checkpoint `ITS` đúng miền đã cải thiện rất mạnh so với lần chạy trước.

Trên SOTS-Outdoor, MB-TaylorFormerV2 cao hơn UDPNet `0.13 dB` và `0.0002` SSIM, nhưng chậm hơn khoảng `26.42 ms/ảnh`. Trên O-HAZE, hai mô hình mới gần như tương đương: UDPNet hơn `0.04 dB`, còn MB-TaylorFormerV2 hơn `0.0012` SSIM.

Trên I-HAZE, UDPNet hơn MB-TaylorFormerV2 `0.85 dB` PSNR và `0.0472` SSIM, nhưng vẫn thấp hơn GridDehazeNet `2.46 dB` PSNR và `0.1985` SSIM. Checkpoint ITS giúp cải thiện SOTS-Indoor nhưng không bảo đảm tối ưu cho I-HAZE, vì I-HAZE là ảnh real có phân phối khác đáng kể.

## 2. Sheet Real / Synthetic

| Mô hình | Real PSNR / SSIM / ms | Synthetic PSNR / SSIM / ms |
|---|---:|---:|
| UDPNet | **15.64 / 0.6585 / 185.85** | **36.40 / 0.9898 / 144.45** |
| MB-TaylorFormerV2 | 15.26 / 0.6395 / 287.84 | 35.50 / 0.9876 / 173.35 |
| GridDehazeNet | **16.10 / 0.6878 / 1222.96** | 32.34 / 0.9850 / **44.84** |
| CAP | 16.08 / **0.6860 / 1400.07** | 22.10 / 0.9022 / **19.17** |
| DCP | 13.15 / 0.5790 / 2548.15 | 19.40 / 0.8769 / 46.03 |

UDPNet dẫn đầu synthetic với `36.40 dB`, cao hơn MB-TaylorFormerV2 `0.90 dB` và GridDehazeNet `4.06 dB`. Trên real, GridDehazeNet vẫn có điểm cao nhất nhưng chậm `1222.96 ms/ảnh`; UDPNet thấp hơn `0.46 dB` nhưng nhanh hơn khoảng `6.6 lần`. Khoảng cách synthetic-real của UDPNet là `20.76 dB`, cho thấy domain gap rất lớn dù đã dùng checkpoint theo miền tổng hợp.

## 3. Sheet Fog level

| Mô hình | Light fog PSNR / SSIM / ms | Medium fog PSNR / SSIM / ms | Heavy fog PSNR / SSIM / ms |
|---|---:|---:|---:|
| UDPNet | **37.66 / 0.9917 / 144.91** | **36.51 / 0.9897 / 136.13** | **34.01 / 0.9866 / 152.65** |
| MB-TaylorFormerV2 | 37.13 / 0.9912 / 175.26 | 34.97 / 0.9854 / 166.18 | 33.14 / 0.9837 / 177.67 |
| GridDehazeNet | 33.79 / 0.9882 / **44.81** | 32.59 / 0.9847 / **33.49** | 29.46 / 0.9794 / **57.21** |
| CAP | 22.85 / 0.9097 / **18.59** | 21.74 / 0.8897 / **19.55** | 21.14 / 0.9021 / **19.82** |
| DCP | 19.56 / 0.8897 / 44.29 | 19.75 / 0.8678 / 47.32 | 18.74 / 0.8637 / 47.77 |

UDPNet dẫn đầu cả ba mức haze. So với MB-TaylorFormerV2, UDPNet hơn `0.53 dB` ở haze nhẹ, `1.54 dB` ở haze trung bình và `0.87 dB` ở haze nặng. GridDehazeNet nhanh hơn nhưng thấp hơn UDPNet lần lượt `3.87`, `3.92` và `4.55 dB`. Fog level chỉ áp dụng cho synthetic và được suy ra từ metadata/tên ảnh.

## 4. Sheet Hardware

| Mô hình | RAM peak (GB) | FPS toàn pipeline | FPS suy luận | VRAM peak (GB / 80 GB) |
|---|---:|---:|---:|---:|
| UDPNet | 1.941 | **3.6700** | **6.7695** | 46.373 |
| MB-TaylorFormerV2 | 1.751 | 3.0658 | 3.0658 | 30.846 |
| GridDehazeNet | 1.790 | 2.3004 | 7.6970 | 19.290 |
| CAP | 1.190 | 1.8623 | 4.1056 | 0 |
| DCP | 1.290 | 2.3988 | 7.8008 | 0 |

UDPNet nhanh hơn MB-TaylorFormerV2 khoảng `20%` theo FPS toàn pipeline. UDPNet dùng `46.373 GB` VRAM do chạy đồng thời Depth Anything V2 và UDPNet; MB-TaylorFormerV2 dùng `30.846 GB` nhưng chậm hơn. GridDehazeNet có FPS model cao nhất nhưng pipeline tổng thể chậm do xử lý ảnh thực tế và post-processing. VRAM bằng `0` của CAP/DCP chỉ có nghĩa evaluator không ghi nhận CUDA memory vì hai phương pháp chạy CPU.

## 5. Kết luận

Sau khi dùng checkpoint theo miền, UDPNet cải thiện mạnh trên SOTS-Indoor và dẫn đầu cả ba mức fog synthetic. UDPNet là lựa chọn cân bằng tốt nhất giữa chất lượng và tốc độ trong hai mô hình mới. MB-TaylorFormerV2 gần UDPNet trên SOTS-Outdoor và O-HAZE, nhưng chậm hơn và thường có PSNR thấp hơn.

GridDehazeNet vẫn mạnh nhất trên I-HAZE và có tốc độ infer model cao, nhưng pipeline thực tế rất chậm trên ảnh độ phân giải lớn. Kết quả I-HAZE thấp của UDPNet/MB-TaylorFormerV2 dù dùng checkpoint ITS cho thấy đúng checkpoint theo tên dataset chưa đủ; preprocessing, domain real và việc resize `max-side=512` vẫn ảnh hưởng lớn. Muốn tái lập sát paper cần dùng đúng preprocessing/checkpoint validation và inference full-resolution hoặc tiled.

## 6. Miền tần số trong HazeWaveNet

`paper/Dehaze wavelet.pdf` mô tả HazeWaveNet, không phải UDPNet/MB-TaylorFormerV2. Mạng dùng DWT Haar nhiều mức để tách đặc trưng thành `LL` (thấp tần) và `LH/HL/HH` (cao tần). Nhánh `LL` được xử lý mạnh bằng haze-trend guidance từ Dark Channel Prior để sửa veil haze, tương phản và vùng trơn. Các dải cao tần được xử lý nhẹ để giữ biên và texture. Wavelet Integration Module học cách hợp nhất các dải bằng convolution, residual và upsampling thay vì dùng trực tiếp IDWT. Loss gồm amplitude loss và decomposition loss trên các dải DWT, buộc ảnh đầu ra nhất quán cả ở miền pixel và miền tần số.
