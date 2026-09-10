# Chuẩn bị MIT-BIH Arrhythmia Database

Dataset không được commit vào repository. Pipeline cần mỗi record có đủ ba file
`.dat`, `.hea` và `.atr` trong cùng một thư mục.

## Tải bằng WFDB

Từ thư mục gốc repository:

```bash
python -c "import wfdb; wfdb.dl_database('mitdb', dl_dir='data/mitdb')"
```

Hoặc tải thủ công từ [PhysioNet MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/).

Kiểm tra nhanh:

```bash
python -c "from pathlib import Path; p=Path('data/mitdb'); print(len(list(p.glob('*.hea'))), 'records')"
```

Bộ MIT-BIH đầy đủ có 48 record. Sau khi tải xong, chạy:

```powershell
$env:MITBIH_DATASET_PATH = (Resolve-Path "data\mitdb").Path
python src/mixed_precision.py
```

## Thiết lập thí nghiệm hiện tại

| Hạng mục | Giá trị |
|---|---|
| Độ dài nhịp | 260 mẫu |
| Chuẩn hóa | Z-score theo từng nhịp |
| Nhóm AAMI | `F`, `N`, `Q`, `S`, `V` |
| Split checkpoint công bố | Random beat-level, không patient-disjoint |
| Cân bằng | Không SMOTE; class weight `[1.5, 1, 1, 1, 1]` |

Khi làm nghiên cứu nghiêm túc, hãy thay bằng patient-disjoint split và lưu danh
sách record của từng tập trong báo cáo để người khác có thể tái lập.
