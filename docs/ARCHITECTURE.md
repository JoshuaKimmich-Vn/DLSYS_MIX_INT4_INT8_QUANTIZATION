# Kiến trúc và luồng xử lý

## Pipeline

```text
MIT-BIH records (.dat/.hea/.atr)
        |
        v
BeatExtractor -> cửa sổ 260 mẫu -> Z-score từng nhịp
        |
        v
PaperInceptionCNN (DSC Conv1d + Inception blocks, 1.658 tham số)
        |
        v
Sensitivity leave-one-out
        |
        +--> 40% layer ít nhạy: weight INT4
        `--> 60% layer còn lại: weight INT8
        |
        v
FakeQuant QAT -> hiệu chuẩn observer bằng train set
        |
        v
RealQuant Python reference -> metric + ước tính packed size
```

## Quyết định thiết kế

- Depthwise-separable convolution giảm số phép tính và tham số.
- Không dùng BatchNorm để giảm trạng thái cần fold khi triển khai nhúng.
- Weight dùng per-channel symmetric quantization; activation dùng per-tensor
  asymmetric quantization.
- Bit-width được chọn từ `config/int8_sensitivity.json`.
- Observer chỉ hiệu chuẩn bằng tập train rồi đóng băng trước khi đánh giá.

## Giới hạn cần hiểu đúng

`RealQuantConv1d` và `RealQuantLinear` lưu integer weight thật nhưng dequantize
về số thực trước toán tử PyTorch. Đây là mô hình tham chiếu để kiểm tra miền bit
và sai số; chưa phải kernel integer-only. Với INT4, tensor tham chiếu vẫn dùng
một byte cho mỗi phần tử. Con số 3.587 byte là ước tính gồm packed weight,
quantization metadata và FP32 bias.
