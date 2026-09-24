BÁO CÁO KẾT QUẢ THỰC NGHIỆM CÁC GIẢI PHÁP DEHAZING

Phạm vi đánh giá

Các mô hình được đánh giá trên bốn bộ dữ liệu: SOTS-Indoor, SOTS-Outdoor, O-HAZE và I-HAZE. Các giải pháp gồm UDPNet, MB-TaylorFormerV2, GridDehazeNet, CAP và DCP.

UDPNet sử dụng Depth Anything V2 kết hợp với UDPNet. MB-TaylorFormerV2 sử dụng checkpoint B. UDPNet và MB-TaylorFormerV2 sử dụng checkpoint theo miền: checkpoint ITS cho SOTS-Indoor và I-HAZE, checkpoint OTS cho SOTS-Outdoor và O-HAZE. Ảnh đầu vào được giới hạn max-side=512 để phù hợp với VRAM Colab. Các kết quả dưới đây chỉ mô tả lần chạy hiện tại.

1. SHEET DATASET

Số liệu có dạng PSNR / SSIM / thời gian trung bình trên mỗi ảnh:

Giải pháp              SOTS-Indoor          SOTS-Outdoor         O-HAZE               I-HAZE
                       PSNR SSIM ms         PSNR SSIM ms         PSNR SSIM ms         PSNR SSIM ms
UDPNet                 37.54 0.9919 119.77  35.26 0.9877 169.14  18.27 0.7540 159.99  11.94 0.5247 222.04
MB-TaylorFormerV2      35.61 0.9873 151.13  35.39 0.9879 195.56  18.23 0.7552 262.02  11.09 0.4775 323.99
GridDehazeNet          33.21 0.9860  42.17  31.47 0.9839  47.52  17.32 0.6625 1126.20 14.40 0.7232 1358.43
CAP                    22.76 0.8833  19.59  21.44 0.9211  18.75  16.78 0.6511 1264.76 15.12 0.7348 1589.52
DCP                    21.70 0.8915  47.87  17.10 0.8623  44.19  14.30 0.5722 2335.65 11.54 0.5886 2845.65

SOTS-Indoor

UDPNet đạt 37.54 dB, cao hơn MB-TaylorFormerV2 1.93 dB và cao hơn GridDehazeNet 4.33 dB. SSIM của UDPNet cũng cao nhất, đạt 0.9919. Kết quả cho thấy checkpoint ITS phù hợp với miền indoor và UDPNet khôi phục tốt cả chất lượng pixel lẫn cấu trúc.

MB-TaylorFormerV2 đạt 35.61 dB và 0.9873 SSIM, thấp hơn UDPNet nhưng vẫn cao hơn GridDehazeNet. Thời gian 151.13 ms/ảnh chậm hơn UDPNet.

GridDehazeNet có tốc độ nhanh nhất trong các mô hình học sâu, nhưng PSNR thấp hơn UDPNet. CAP và DCP có thời gian thấp trong dataset này, nhưng chất lượng ảnh thấp hơn rõ rệt.

SOTS-Outdoor

MB-TaylorFormerV2 đạt PSNR cao nhất với 35.39 dB. UDPNet đạt 35.26 dB, chỉ thấp hơn 0.13 dB. MB-TaylorFormerV2 có SSIM 0.9879, cao hơn UDPNet 0.0002.

Hai mô hình đều vượt GridDehazeNet hơn 3.7 dB PSNR. UDPNet nhanh hơn MB-TaylorFormerV2 26.42 ms/ảnh. UDPNet có ưu thế về tốc độ, còn MB-TaylorFormerV2 có chất lượng gần tương đương.

O-HAZE

UDPNet và MB-TaylorFormerV2 gần như ngang nhau. UDPNet đạt 18.27 dB, MB-TaylorFormerV2 đạt 18.23 dB. Chênh lệch 0.04 dB không đáng kể. MB-TaylorFormerV2 có SSIM cao hơn rất nhỏ, 0.7552 so với 0.7540.

Cả hai mô hình học sâu đều cao hơn GridDehazeNet về PSNR và SSIM, đồng thời nhanh hơn rất nhiều. GridDehazeNet mất 1126.20 ms/ảnh, trong khi UDPNet mất 159.99 ms/ảnh.

I-HAZE

UDPNet đạt 11.94 dB và 0.5247 SSIM, cao hơn MB-TaylorFormerV2 với 11.09 dB và 0.4775 SSIM. Tuy nhiên GridDehazeNet có PSNR 14.40 dB và SSIM 0.7232, cao hơn cả hai mô hình.

Kết quả cho thấy checkpoint ITS không tự động bảo đảm kết quả cao trên I-HAZE. I-HAZE là bộ ảnh thực có phân phối haze, màu sắc, ánh sáng và độ phân giải khác SOTS-Indoor. Preprocessing và việc resize max-side=512 cũng có thể làm mất nhiều chi tiết.

2. SHEET REAL VÀ SYNTHETIC

Giải pháp              Real                  Synthetic
                       PSNR SSIM ms          PSNR SSIM ms
UDPNet                 15.64 0.6585 185.85   36.40 0.9898 144.45
MB-TaylorFormerV2      15.26 0.6395 287.84   35.50 0.9876 173.35
GridDehazeNet          16.10 0.6878 1222.96  32.34 0.9850 44.84
CAP                    16.08 0.6860 1400.07  22.10 0.9022 19.17
DCP                    13.15 0.5790 2548.15  19.40 0.8769 46.03

Trên miền synthetic, UDPNet đứng đầu với 36.40 dB, cao hơn MB-TaylorFormerV2 0.90 dB và GridDehazeNet 4.06 dB. Điều này phù hợp với việc UDPNet dùng checkpoint ITS cho SOTS-Indoor và OTS cho SOTS-Outdoor.

Trên miền real, GridDehazeNet có điểm cao nhất nhưng thời gian 1222.96 ms/ảnh rất lớn. UDPNet thấp hơn GridDehazeNet 0.46 dB nhưng nhanh hơn khoảng 6.6 lần. MB-TaylorFormerV2 thấp hơn UDPNet 0.38 dB và chậm hơn 101.99 ms/ảnh.

Khoảng cách giữa synthetic và real của UDPNet là 20.76 dB. Đây là domain gap lớn, cho thấy model được tối ưu tốt cho ảnh haze tổng hợp nhưng khả năng tổng quát sang ảnh thực vẫn bị giới hạn.

3. SHEET FOG LEVEL

Giải pháp              Light fog             Medium fog           Heavy fog
                       PSNR SSIM ms         PSNR SSIM ms         PSNR SSIM ms
UDPNet                 37.66 0.9917 144.91  36.51 0.9897 136.13  34.01 0.9866 152.65
MB-TaylorFormerV2      37.13 0.9912 175.26  34.97 0.9854 166.18  33.14 0.9837 177.67
GridDehazeNet          33.79 0.9882  44.81  32.59 0.9847  33.49  29.46 0.9794  57.21
CAP                    22.85 0.9097  18.59  21.74 0.8897  19.55  21.14 0.9021  19.82
DCP                    19.56 0.8897  44.29  19.75 0.8678  47.32  18.74 0.8637  47.77

UDPNet đứng đầu cả ba mức haze. So với MB-TaylorFormerV2, UDPNet cao hơn 0.53 dB ở haze nhẹ, 1.54 dB ở haze trung bình và 0.87 dB ở haze nặng.

GridDehazeNet nhanh hơn nhiều, nhưng thấp hơn UDPNet lần lượt 3.87, 3.92 và 4.55 dB. Fog level chỉ áp dụng cho dữ liệu synthetic và được suy ra từ metadata hoặc tên ảnh, không phải phép đo trực tiếp mật độ haze vật lý.

4. SHEET HARDWARE

Giải pháp              RAM peak   FPS toàn pipeline   FPS suy luận   VRAM peak / 80 GB
UDPNet                 1.941 GB   3.6700              6.7695         46.373 GB
MB-TaylorFormerV2      1.751 GB   3.0658              3.0658         30.846 GB
GridDehazeNet          1.790 GB   2.3004              7.6970         19.290 GB
CAP                    1.190 GB   1.8623              4.1056          0 GB
DCP                    1.290 GB   2.3988              7.8008          0 GB

UDPNet nhanh hơn MB-TaylorFormerV2 khoảng 20 phần trăm theo FPS toàn pipeline. UDPNet dùng 46.373 GB VRAM vì phải chạy cả Depth Anything V2 và UDPNet. MB-TaylorFormerV2 dùng 30.846 GB nhưng FPS suy luận chỉ đạt 3.0658.

GridDehazeNet có FPS suy luận cao nhất, 7.6970, và chỉ dùng 19.290 GB VRAM. Tuy nhiên FPS toàn pipeline chỉ 2.3004 do chi phí đọc ảnh, xử lý ảnh và lưu kết quả, đặc biệt với ảnh thực có độ phân giải lớn.

Giá trị VRAM bằng 0 của CAP và DCP không có nghĩa là hai phương pháp không dùng bộ nhớ. Hai phương pháp này chạy CPU nên evaluator không ghi nhận CUDA memory.

5. KẾT LUẬN

UDPNet là giải pháp cân bằng nhất giữa chất lượng và tốc độ trong cấu hình hiện tại. UDPNet đứng đầu SOTS-Indoor, O-HAZE và cả ba mức fog synthetic. MB-TaylorFormerV2 gần UDPNet trên SOTS-Outdoor và O-HAZE, nhưng chậm hơn.

GridDehazeNet có kết quả tốt nhất trên I-HAZE và FPS infer model cao, nhưng pipeline thực tế chậm hơn rất nhiều. CAP và DCP phù hợp làm baseline tốc độ hoặc phương pháp truyền thống, không phù hợp nếu ưu tiên chất lượng.

Kết quả I-HAZE của UDPNet và MB-TaylorFormerV2 vẫn thấp dù dùng checkpoint ITS. Nguyên nhân có thể đến từ domain thực khác biệt, preprocessing, checkpoint và việc resize max-side=512. Muốn tái lập sát paper cần dùng đúng preprocessing, checkpoint được chọn trên validation tương ứng và inference full-resolution hoặc tiled.

6. GIẢI THÍCH MIỀN TẦN SỐ TRONG HAZEWAVENET

Paper Dehaze wavelet mô tả HazeWaveNet, không phải UDPNet hay MB-TaylorFormerV2. Mạng dùng biến đổi wavelet rời rạc DWT Haar nhiều mức để tách đặc trưng thành LL, LH, HL và HH.

LL là thành phần thấp tần, chứa vùng trơn, cấu trúc lớn, độ sáng và tương phản tổng thể. LH, HL và HH là các thành phần cao tần, chứa biên, texture và các biến thiên nhanh.

Nhánh LL được xử lý mạnh bằng haze-trend guidance từ Dark Channel Prior để sửa veil haze, tương phản và vùng mờ. Các dải cao tần được xử lý nhẹ để giữ biên và texture, tránh làm ảnh bị nhòe.

Wavelet Integration Module hợp nhất các dải tần bằng convolution, residual connection và upsampling học được. Paper không dùng trực tiếp IDWT ở bước cuối; WIM học cách tái tạo để giảm ringing và checkerboard artifact.

Hàm mất mát gồm amplitude loss và decomposition loss trên các dải LL, LH, HL, HH. Ràng buộc này giúp ảnh đầu ra vừa đúng ở miền pixel vừa nhất quán về cấu trúc tần số.
