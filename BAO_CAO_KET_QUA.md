BAO CAO KET QUA THUC NGHIEM CAC GIAI PHAP DEHAZING

Pham vi danh gia

Nam giai phap duoc danh gia tren bon bo du lieu SOTS-Indoor, SOTS-Outdoor, O-HAZE va I-HAZE: UDPNet, MB-TaylorFormerV2, GridDehazeNet, CAP va DCP.

UDPNet su dung Depth Anything V2 ket hop voi UDPNet. MB-TaylorFormerV2 su dung checkpoint B. UDPNet va MB-TaylorFormerV2 su dung checkpoint theo mien: checkpoint ITS cho SOTS-Indoor va I-HAZE, checkpoint OTS cho SOTS-Outdoor va O-HAZE. Anh dau vao duoc gioi han max-side=512 de phu hop voi VRAM Colab. Cac ket qua duoi day chi mo ta mot lan chay hien tai.

1. SHEET DATASET

So lieu PSNR, SSIM va thoi gian trung binh tren moi anh:

Giai phap              SOTS-Indoor          SOTS-Outdoor         O-HAZE               I-HAZE
                       PSNR SSIM ms         PSNR SSIM ms         PSNR SSIM ms         PSNR SSIM ms
UDPNet                 37.54 0.9919 119.77  35.26 0.9877 169.14  18.27 0.7540 159.99  11.94 0.5247 222.04
MB-TaylorFormerV2      35.61 0.9873 151.13  35.39 0.9879 195.56  18.23 0.7552 262.02  11.09 0.4775 323.99
GridDehazeNet          33.21 0.9860  42.17  31.47 0.9839  47.52  17.32 0.6625 1126.20 14.40 0.7232 1358.43
CAP                    22.76 0.8833  19.59  21.44 0.9211  18.75  16.78 0.6511 1264.76 15.12 0.7348 1589.52
DCP                    21.70 0.8915  47.87  17.10 0.8623  44.19  14.30 0.5722 2335.65 11.54 0.5886 2845.65

SOTS-Indoor

UDPNet dat 37.54 dB, cao hon MB-TaylorFormerV2 1.93 dB va cao hon GridDehazeNet 4.33 dB. SSIM cua UDPNet cung cao nhat, dat 0.9919. Ket qua nay cho thay checkpoint ITS phu hop voi mien indoor va UDPNet phuc hoi tot ca chat luong pixel lan cau truc.

MB-TaylorFormerV2 dat 35.61 dB va 0.9873 SSIM, thap hon UDPNet nhung van cao hon GridDehazeNet. Thoi gian 151.13 ms/anh cham hon UDPNet.

GridDehazeNet co toc do nhanh nhat trong cac mo hinh hoc sau, nhung PSNR thap hon UDPNet. CAP va DCP co toc do thap nhat ve ms/anh trong dataset nay, nhung chat luong anh thap hon nhieu.

SOTS-Outdoor

MB-TaylorFormerV2 dat PSNR cao nhat voi 35.39 dB. UDPNet dat 35.26 dB, chi thap hon 0.13 dB. MB-TaylorFormerV2 co SSIM 0.9879, cao hon UDPNet 0.0002.

Hai mo hinh deu vuot GridDehazeNet hon 3.7 dB PSNR. UDPNet nhanh hon MB-TaylorFormerV2 26.42 ms/anh. Dieu nay cho thay UDPNet co uu the ve toc do, con MB-TaylorFormerV2 co chat luong gan tuong duong.

O-HAZE

UDPNet va MB-TaylorFormerV2 gan nhu ngang nhau. UDPNet dat 18.27 dB, MB-TaylorFormerV2 dat 18.23 dB. Chenh lech 0.04 dB khong dang ke. MB-TaylorFormerV2 co SSIM cao hon rat nho, 0.7552 so voi 0.7540.

Ca hai mo hinh hoc sau deu cao hon GridDehazeNet ve PSNR va SSIM, dong thoi nhanh hon rat nhieu. GridDehazeNet mat 1126.20 ms/anh, trong khi UDPNet mat 159.99 ms/anh.

I-HAZE

UDPNet dat 11.94 dB va 0.5247 SSIM, cao hon MB-TaylorFormerV2 voi 11.09 dB va 0.4775 SSIM. Tuy nhien GridDehazeNet co PSNR 14.40 dB va SSIM 0.7232, cao hon ca hai mo hinh.

Ket qua nay cho thay checkpoint ITS khong tu dong bao dam ket qua cao tren I-HAZE. I-HAZE la bo anh real co phan phoi haze, mau sac, anh sang va do phan giai khac voi SOTS-Indoor. Preprocessing va viec resize max-side=512 cung co the lam mat nhieu chi tiet.

2. SHEET REAL VA SYNTHETIC

Giai phap              Real                  Synthetic
                       PSNR SSIM ms          PSNR SSIM ms
UDPNet                 15.64 0.6585 185.85   36.40 0.9898 144.45
MB-TaylorFormerV2      15.26 0.6395 287.84   35.50 0.9876 173.35
GridDehazeNet          16.10 0.6878 1222.96   32.34 0.9850 44.84
CAP                    16.08 0.6860 1400.07   22.10 0.9022 19.17
DCP                    13.15 0.5790 2548.15   19.40 0.8769 46.03

Tren mien synthetic, UDPNet dung dau voi 36.40 dB, cao hon MB-TaylorFormerV2 0.90 dB va cao hon GridDehazeNet 4.06 dB. Dieu nay phu hop voi viec UDPNet dung checkpoint ITS cho SOTS-Indoor va OTS cho SOTS-Outdoor.

Tren mien real, GridDehazeNet co PSNR va SSIM cao nhat. Tuy nhien thoi gian 1222.96 ms/anh rat lon. UDPNet thap hon GridDehazeNet 0.46 dB nhung nhanh hon khoang 6.6 lan. MB-TaylorFormerV2 thap hon UDPNet 0.38 dB va cham hon 101.99 ms/anh.

Khoang cach giua synthetic va real cua UDPNet la 20.76 dB. Day la domain gap lon, cho thay model duoc toi uu tot cho anh haze tong hop nhung kha nang tong quat sang anh real van bi gioi han.

3. SHEET FOG LEVEL

Giai phap              Light fog             Medium fog           Heavy fog
                       PSNR SSIM ms         PSNR SSIM ms         PSNR SSIM ms
UDPNet                 37.66 0.9917 144.91  36.51 0.9897 136.13  34.01 0.9866 152.65
MB-TaylorFormerV2      37.13 0.9912 175.26  34.97 0.9854 166.18  33.14 0.9837 177.67
GridDehazeNet          33.79 0.9882  44.81  32.59 0.9847  33.49  29.46 0.9794  57.21
CAP                    22.85 0.9097  18.59  21.74 0.8897  19.55  21.14 0.9021  19.82
DCP                    19.56 0.8897  44.29  19.75 0.8678  47.32  18.74 0.8637  47.77

UDPNet dung dau ca ba muc haze. So voi MB-TaylorFormerV2, UDPNet cao hon 0.53 dB o haze nhe, 1.54 dB o haze trung binh va 0.87 dB o haze nang.

GridDehazeNet nhanh hon nhieu, nhung thap hon UDPNet 3.87 dB o haze nhe, 3.92 dB o haze trung binh va 4.55 dB o haze nang. Fog level chi ap dung cho du lieu synthetic va duoc suy ra tu metadata hoac ten anh, khong phai phep do truc tiep mat do haze vat ly.

4. SHEET HARDWARE

Giai phap              RAM peak   FPS toan pipeline   FPS suy luan   VRAM peak / 80 GB
UDPNet                 1.941 GB   3.6700              6.7695         46.373 GB
MB-TaylorFormerV2      1.751 GB   3.0658              3.0658         30.846 GB
GridDehazeNet          1.790 GB   2.3004              7.6970         19.290 GB
CAP                    1.190 GB   1.8623              4.1056          0 GB
DCP                    1.290 GB   2.3988              7.8008          0 GB

UDPNet nhanh hon MB-TaylorFormerV2 khoang 20 phan tram theo FPS toan pipeline. UDPNet dung 46.373 GB VRAM vi phai chay ca Depth Anything V2 va UDPNet. MB-TaylorFormerV2 dung 30.846 GB nhung FPS suy luan chi dat 3.0658.

GridDehazeNet co FPS suy luan cao nhat, 7.6970, va chi dung 19.290 GB VRAM. Tuy nhien FPS toan pipeline chi 2.3004 do chi phi doc anh, xu ly anh va luu ket qua, dac biet voi anh real co do phan giai lon.

Gia tri VRAM bang 0 cua CAP va DCP khong co nghia la hai phuong phap khong dung bo nho. Hai phuong phap nay chay CPU nen evaluator khong ghi nhan CUDA memory.

5. KET LUAN

UDPNet la giai phap can bang nhat giua chat luong va toc do trong cau hinh hien tai. UDPNet dung dau SOTS-Indoor, O-HAZE va ca ba muc fog synthetic. MB-TaylorFormerV2 gan UDPNet tren SOTS-Outdoor va O-HAZE, nhung cham hon.

GridDehazeNet co ket qua tot nhat tren I-HAZE va FPS infer model cao, nhung pipeline thuc te cham hon rat nhieu. CAP va DCP phu hop lam baseline toc do hoac phuong phap truyen thong, khong phu hop neu uu tien chat luong.

Ket qua I-HAZE cua UDPNet va MB-TaylorFormerV2 van thap du dung checkpoint ITS. Nguyen nhan co the den tu domain real khac biet, preprocessing, checkpoint va viec resize max-side=512. Muon tai lap sat paper can dung dung preprocessing, checkpoint duoc chon tren validation tuong ung va inference full-resolution hoac tiled.

6. GIAI THICH MIEN TAN SO TRONG HAZEWAVENET

Paper Dehaze wavelet mo ta HazeWaveNet, khong phai UDPNet hay MB-TaylorFormerV2. Mang dung bien doi wavelet roi rac DWT Haar nhieu muc de tach dac trung thanh LL, LH, HL va HH.

LL la thanh phan thap tan, chua vung tron, cau truc lon, do sang va tuong phan tong the. LH, HL va HH la cac thanh phan cao tan, chua bien, texture va cac bien thien nhanh.

Nhan LL duoc xu ly manh bang haze-trend guidance tu Dark Channel Prior de sua veil haze, tuong phan va vung mo. Cac dai cao tan duoc xu ly nhe de giu bien va texture, tranh lam anh bi nhoe.

Wavelet Integration Module hop nhat cac dai tan bang convolution, residual connection va upsampling hoc duoc. Paper khong dung truc tiep IDWT o buoc cuoi; WIM hoc cach tai tao de giam ringing va checkerboard artifact.

Ham mat mat gom amplitude loss va decomposition loss tren cac dai LL, LH, HL, HH. Nhieu rang buoc nay giup anh dau ra vua dung o mien pixel vua nhat quan ve cau truc tan so.
