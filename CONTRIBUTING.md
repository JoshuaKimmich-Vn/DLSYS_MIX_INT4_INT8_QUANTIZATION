# Đóng góp

Cảm ơn bạn muốn cải thiện dự án. Hãy giữ mỗi pull request tập trung vào một vấn
đề, mô tả rõ thay đổi và cách kiểm chứng.

## Quy trình đề xuất

1. Tạo issue hoặc mô tả mục tiêu kỹ thuật.
2. Tạo branch, ví dụ `feature/int4-packing` hoặc `fix/patient-split`.
3. Không commit dataset, cache, môi trường ảo hay output build.
4. Thêm hoặc cập nhật test nếu hành vi của code thay đổi.
5. Chạy `python -m unittest discover -s tests -v` trước khi mở pull request.
6. Ghi rõ dataset split, seed, metric và phần cứng nếu công bố kết quả mới.

## Quy ước kỹ thuật

- Dùng Python 3.10+ và UTF-8.
- Ưu tiên đường dẫn tương đối; không đưa đường dẫn máy cá nhân vào code.
- Không gọi mô hình là integer-only nếu toán tử vẫn dequantize về FP32.
- Phân biệt kích thước tensor Python với kích thước packed khi triển khai.
- Kết quả y sinh phải nêu rõ cách chia bệnh nhân và giới hạn sử dụng.
