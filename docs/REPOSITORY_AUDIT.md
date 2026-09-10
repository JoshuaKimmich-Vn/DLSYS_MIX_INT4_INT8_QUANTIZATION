# Báo cáo rà soát repository

Ngày rà soát: 2026-09-10

## Kết quả

| Hạng mục | Trạng thái |
|---|---|
| Cấu trúc `src/tests/models/results/data/docs` | Đạt |
| Đường dẫn tuyệt đối của máy cá nhân trong code | Đã loại bỏ |
| Dependency trùng lặp | Đã gộp về `requirements.txt` |
| License được README công bố | Đã bổ sung MIT License |
| Dataset có nguy cơ bị upload | Đã chặn bằng `.gitignore` |
| Checkpoint vượt giới hạn GitHub 100 MB | Không; file lớn nhất khoảng 189 KB |
| Python compile check | Đạt |
| Regression test | 4/4 đạt |
| GitHub Actions | Đã cấu hình |

## Nội dung được giữ để tái lập

- checkpoint FP32 và mixed-QAT;
- báo cáo sensitivity dùng để phân phối INT4/INT8;
- báo cáo JSON/TXT tham chiếu;
- mã mô hình, lượng tử hóa và kiểm thử;
- hướng dẫn dữ liệu, kiến trúc, đóng góp và portfolio.

## Nội dung không đưa lên GitHub

- `data/mitdb/`;
- cache Python, môi trường ảo và file IDE;
- báo cáo chạy lại trong `results/generated/`;
- binary build cục bộ.

## Vấn đề dữ liệu cục bộ

Thư mục `data/mitdb/` tại thời điểm rà soát chỉ có 9 header, 8 signal và 7
annotation; record `104` và `109` chưa đủ bộ ba file. Đây không ảnh hưởng các
regression test nhưng phải tải đủ dataset trước khi chạy đánh giá end-to-end.

## Giới hạn khoa học còn lại

- Checkpoint công bố dùng random beat-level split, chưa patient-disjoint.
- INT4 mới là miền giá trị; chưa có nibble packing và kernel C integer-only.
- Kích thước packed là ước tính, chưa phải số đo Flash/RAM trên phần cứng.
