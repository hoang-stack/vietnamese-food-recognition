# Results

| Model | Top-1 (95% CI) | Top-5 | Macro-F1 | ECE raw → scaled | Top-1, clean test photos |
| --- | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt-Tiny, fine-tuned | 91.2% (90.4–92.0) | 98.2% | 0.913 | 0.074 → 0.018 | 91.1% |
| CLIP ViT-B/16, linear probe | 89.3% (88.4–90.1) | 98.8% | 0.894 | 0.076 → 0.015 | 88.8% |
| EfficientNet-B0, fine-tuned | 89.0% (88.2–89.8) | 97.7% | 0.889 | 0.087 → 0.012 | 88.7% |
| CLIP ViT-B/16, zero-shot | 52.7% (51.2–54.0) | 86.4% | 0.503 | 0.035 → 0.023 | 51.8% |

## Leakage

439 of 5040 test photos (8.7%) have a near-duplicate in the training set (64 with an identical perceptual hash); 91.3% of those duplicates carry the same label.

## Most frequent confusions (best model)

| True | Predicted | Count |
| --- | --- | ---: |
| Bánh canh | Hủ tiếu | 15 |
| Bún riêu | Bún bò Huế | 13 |
| Bún bò Huế | Phở | 12 |
| Phở | Bún bò Huế | 12 |
| Hủ tiếu | Phở | 10 |
| Bún bò Huế | Bún riêu | 9 |
| Hủ tiếu | Bánh canh | 9 |
| Phở | Hủ tiếu | 9 |
| Cao lầu | Mì Quảng | 8 |
| Bánh canh | Bún mắm | 7 |
| Bánh xèo | Bánh tráng nướng | 7 |
| Bánh đúc | Bánh cuốn | 6 |
| Bánh khọt | Bánh căn | 6 |
| Bánh tét | Bánh chưng | 6 |
| Bún bò Huế | Bánh canh | 6 |

## Label-noise candidates per class (top 10)

| Dish | Share |
| --- | ---: |
| Bánh canh | 62.3% |
| Hủ tiếu | 51.3% |
| Bánh đúc | 46.4% |
| Nem chua | 33.2% |
| Bún riêu | 30.3% |
| Phở | 29.4% |
| Bánh bột lọc | 28.8% |
| Mì Quảng | 27.8% |
| Bún mắm | 20.9% |
| Bánh cuốn | 19.2% |
