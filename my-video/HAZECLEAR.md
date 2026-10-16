# Video minh họa Viettel HazeClear

Composition: `HazeClearExplainer`, 74 giây, 1280 x 720, 30 fps. Bản này dùng chữ tiếng Việt, không có giọng đọc hoặc nhạc nền. Animation `MyComp` cũ được giữ nguyên.

## Xem và xuất

```powershell
npm run dev
```

Chọn `HazeClearExplainer` hoặc mở `http://localhost:3000/HazeClearExplainer` nếu Studio đang chạy tại cổng 3000. Mỗi cảnh có timeline riêng trong folder `HazeClear-Scenes`.

Khi muốn xuất MP4:

```powershell
npx remotion render HazeClearExplainer out/Viettel_HazeClear_Explainer.mp4 --codec=h264
```

## Kịch bản

| Thời gian  | Nội dung                                                     |
| ---------- | ------------------------------------------------------------ |
| 0–8 giây   | So sánh ảnh có sương và ảnh khử sương, giới thiệu giải pháp  |
| 8–18 giây  | Tách Haar DWT thành LL, LH, HL, HH                           |
| 18–30 giây | Minh họa quét bốn hướng, chọn lọc thông tin trong Mamba/LFRN |
| 30–38 giây | IWT tái dựng ảnh thô                                         |
| 38–48 giây | Profiling cho thấy DEN là điểm nghẽn suy luận                |
| 48–62 giây | Thay DEN bằng DSRB và so sánh kết quả trên SOTS-OTS          |
| 62–74 giây | Tổng kết trade-off tốc độ–chất lượng và hướng ứng dụng       |

## Nguồn và giới hạn minh họa

- Hazy/clear: WDMamba, Hình 9 trang 10, hàng thứ ba, cột Hazy và Ours. Ảnh từ paper, không phải ảnh mới chạy bằng video này.
- Kiến trúc: Jie Sun và cộng sự, _WDMamba: When Wavelet Degradation Prior Meets Vision Mamba for Image Dehazing_, arXiv:2505.04369, 2025.
- Cơ chế chọn lọc: Albert Gu và Tri Dao, _Mamba: Linear-Time Sequence Modeling with Selective State Spaces_, arXiv:2312.00752v2, 2024.
- Dải Wavelet được tính từ ảnh minh họa bằng tổng/sai phân Haar trên khối 2 x 2. LL hiển thị trung bình; dải chi tiết có offset 128 và tăng tương phản. Nhãn LH/HL theo quy ước tổng/sai phân hai trục, không phải tensor trích từ checkpoint.
- Cảnh quét là sơ đồ nguyên lý, không phải attention map. Ảnh thô làm mờ và hiệu ứng làm nét là mô phỏng giải thích, không phải đầu ra trung gian đo được.
- Luồng tối ưu trong video: DWT, LFRN dùng Mamba, IWT, DSRB, ảnh RGB. Kiến trúc gốc dùng DEN; contrastive regularization chỉ dùng khi huấn luyện.
- Profiling DEN: 24,9351 ms/ảnh và 71,71% thời gian theo khối trên RESIDE-6K, 1.000 ảnh, A100 80 GB, FP32, batch 100, max side 512.
- So sánh SOTS-OTS: baseline 32,35 dB, 35,66 ms/ảnh, 28,0426 FPS; DSRB 31,18 dB, 26,84 ms/ảnh, 37,26 FPS. DSRB tăng 32,87% FPS nhưng giảm 1,17 dB PSNR.
- Viettel HazeClear là giải pháp khử sương hướng tới tiền xử lý cho camera và hệ thống thị giác máy tính. Chưa khẳng định cải thiện detection hoặc đã chạy video camera thời gian thực.

Các hình cần thiết nằm trong `public/hazeclear`; không phụ thuộc đường dẫn ngoài project khi preview hoặc render.
