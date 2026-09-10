# Đưa dự án lên GitHub và CV

## Trước khi public

- Thay `<URL_REPOSITORY_CUA_BAN>` trong README bằng URL thật.
- Cập nhật dòng copyright trong `LICENSE` nếu muốn ghi tên tác giả.
- Chạy test trên máy sạch và kiểm tra GitHub Actions chuyển sang màu xanh.
- Kiểm tra `git status` để chắc chắn `data/mitdb/`, `.venv/` và output sinh ra
  không được stage.
- Chỉ công bố metric có file cấu hình/kết quả đi kèm.
- Thêm ảnh phần cứng, latency hoặc RAM/Flash khi đã có phép đo thực nghiệm.

## Gợi ý mô tả trong CV

> Xây dựng pipeline QAT hỗn hợp INT4/INT8 cho CNN 1D phân loại 5 nhóm nhịp ECG;
> chọn 40% layer INT4 bằng sensitivity analysis, đạt 97,76% accuracy trên split
> beat-level và giảm kích thước packed ước tính 1,85x so với FP32. Viết regression
> test kiểm tra miền lượng tử, đóng băng observer và parity fake/real quant.

Hãy sửa câu trên đúng với đóng góp cá nhân. Nếu chưa tự tái lập thí nghiệm, nên
ghi “reproduced/extended” thay vì nhận toàn bộ pipeline là thiết kế của mình.

## Điểm nên trình bày khi phỏng vấn

1. Vì sao weight INT4 nhưng activation vẫn INT8.
2. Per-channel khác per-tensor quantization thế nào.
3. Vì sao random beat split có thể làm kết quả lạc quan.
4. Vì sao tensor `int8` chứa giá trị INT4 chưa đồng nghĩa tiết kiệm 50% RAM.
5. Cách kiểm tra bit-exact khi chuyển sang C hoặc vi điều khiển.
