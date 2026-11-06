# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM


- **MSSV:**2A202603663
- **Họ và tên:**Lê Quang Trường
- **SEED** (in ở ô đầu notebook):8530

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| `day01_lab.ipynb` — notebook đã chạy, còn output | ☑ |
| `lab1_report.json` — do ô "Báo cáo cuối" sinh ra (15/15 kiểm tra đạt) | ☑ |
| `hand_calc.jpg` — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

| Phương pháp | Precision | Recall | F1 test | F1 SUB 40 | p95 (ms/tin) | USD/1k tin |
|---|---|---|---|---|---|---|
| 1. BoW + LogReg (tự code) | 0.986 | 0.785 | 0.874 | 0.889 | 0.08 | 0 |
| 2. BoW + MLP | 0.988 | 0.860 | 0.920 | 0.919 | 0.18 | 0 |
| 3a. Encoder (MiniLM) + kNN | 0.885 | 0.914 | 0.899 | 0.974 | 67.4 | 0 |
| 3b. Encoder (MiniLM) + LogReg | 0.977 | 0.925 | 0.950 | 0.974 | 58.5 | 0 |
| 3c. Encoder (MiniLM) + MLP head | 0.977 | 0.914 | 0.944 | 0.947 | 62.2 | 0 |
| 4a. LLM nhỏ zero-shot | 0.800 | 1.000 | 0.889 | 0.889 | 2328.6 | 0 |
| 4b. LLM lớn zero-shot | 0.833 | 1.000 | 0.909 | 0.909 | 2355.2 | 0 |
| 4c. LLM nhỏ few-shot | 0.909 | 1.000 | 0.952 | 0.952 | 2471.7 | 0 |

Ghi chú: USD = 0 vì mô hình Groq không có trong bảng giá của notebook, không phải miễn phí hoàn toàn. Các hàng LLM chỉ đo trên SUB (40 tin), nên chênh nhau dưới khoảng 0,05 F1 coi như hòa. Thời gian p95 của Encoder dao động giữa các lần chạy trên CPU Colab (lần trước đo 42 đến 45 ms, lần cuối 58 đến 62 ms).

## 3. Dự đoán và thực tế

**(a) Stress test 50 case khó (`hard_eval`).** Em đoán BoW còn 50-60%, Embedding 80-90%. Thực tế BoW ~52%, Embedding chỉ 60-68%. Em cũng đoán LLM khó nhất ở prompt injection, nhưng thực tế LLM yếu nhất ở tin ham trông giống spam (70%) và tin nói về spam (75%). Em lệch vì nghĩ embedding "hiểu nghĩa" nên sẽ bền. Thật ra embedding chỉ bắt chủ đề của câu chứ không bắt ý định lừa đảo, và dữ liệu train không có kiểu tin giả giọng người quen.
**(b) MLP so với LogReg trên BoW (`dl_curve`).** Em đoán MLP có F1 xấp xỉ logistic regression (~0,874). Thực tế MLP đạt 0,920. Recall tăng từ 0,785 lên 0,860 trong khi precision gần như giữ nguyên (0,986 so với 0,988). Em lệch vì đã đánh giá thấp lợi ích của tính phi tuyến: lớp ẩn có ReLU cho phép kết hợp nhiều từ với nhau để bắt thêm tin spam mà LogReg bỏ sót.
**(c) Đánh đổi độ trễ (`tradeoff`).** Em đoán chỉ kNN và LLM nằm bên phải vạch 50 ms. Thực tế trong lần chạy cuối cả Encoder + LogReg (58,5 ms) và Encoder + MLP head (62,2 ms) cũng vượt vạch. Em lệch vì lần chạy đầu đo được 42 đến 45 ms nên tưởng hai cách này an toàn, quên rằng độ trễ của encoder phụ thuộc phần cứng và dao động giữa các lần chạy.

## 4. Phân tích 50 case khó

BoW + LogReg tụt nhiều nhất: F1 từ 0,889 (test thường) xuống ~0,35, và BoW + MLP cũng tương tự (~0,34). Hai nhóm ham đạt 100% chỉ vì mô hình gần như luôn đoán "ham", trong khi spam giọng tự nhiên (0-10%), né từ khoá (0-17%) và thao túng (12-38%) gần như bị bỏ sót hoàn toàn.
Hai case cụ thể:
- **#13 (spam giọng tự nhiên):** *"Hope you're well! Quick reminder that your photos from the event are ready, view & order prints"*. Tin này không có từ khoá spam nào trong vocab nên BoW cho điểm thấp. Ngay cả LLM nhỏ + prompt phòng thủ cũng đoán sai vì tin trông như thông báo dịch vụ hợp lệ.
- **#2 (ham trông như spam):** *"FREE pizza in the office kitchen right now, come quick before it's gone"*. Có "FREE", "now", giục giã nên LLM nhỏ nghiêng về spam dù là tin đồng nghiệp bình thường. Nó đọc đúng chữ nhưng không đủ ngữ cảnh để biết người gửi là ai.

## 5. Thử thách cá nhân

Đề theo SEED 8530 là **đề #5**: viết 4 tin nhắn của riêng em (2 spam né từ khoá, 2 ham có chữ "free"/"win") rồi so dự đoán BoW và Embedding + LogReg.
4 tin em viết: (1) *"Nhan ngay voucher 500k khi bam vao link duoi na, duy nhat hom nay"* (spam), (2) *"Tai nghe chinh hang xa kho giam 90% chi con 99k, nhanh tay nhe"* (spam), (3) *"Ban co muon nhan ve xem phim free toi nay khong, to tang lai"* (ham), (4) *"Chuc mung ban da win giai nhat cuoc thi thiet ke do hoa cap khoa"* (ham).
**Lưu ý trung thực:** ô code trong notebook bị lỗi `NameError` (em dùng tên biến không có trong notebook như `vectorizer`, `clf_bow`), nên chưa chạy được thật. Các dòng "Mô phỏng kết quả" trong output là **dự đoán của em, không phải kết quả chạy thật**.
Dự đoán của em: BoW dùng từ khoá nên có thể nhầm tin ham chứa "free" và "win" thành spam, và bỏ sót tin spam né từ khoá. Embedding hiểu nghĩa cả câu nên xử lý tin ham tốt hơn. Tuy nhiên theo kết quả stress test, em cũng biết Embedding vẫn có thể sai ở những tin giả giọng quen. Em chưa kiểm chứng được dự đoán này bằng kết quả thật.

## 6. Phương án cho sản phẩm thật

Yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày (~116 tin/giây).
Theo bảng kết quả lần cuối, chỉ BoW + LogReg (0,08 ms) và BoW + MLP (0,18 ms) đạt độ trễ. Hai Encoder (58 và 62 ms) vượt nhẹ, kNN 67 ms, còn LLM 2,3 đến 2,5 giây nên không dùng riêng được. BoW + MLP là lựa chọn nhanh nhất có F1 0,92, nhưng trên 50 case khó chỉ còn ~0,34 nên không an toàn nếu đứng một mình.
Em đề xuất thiết kế lai (router): BoW + MLP chấm xác suất p, tin chắc chắn (p < 0,3 hoặc p > 0,7) được quyết định ngay trong vài phần trăm mili-giây. Tin lưỡng lự (p trong [0,3; 0,7]) chuyển sang Encoder + LogReg hoặc LLM xử lý bất đồng bộ. Nếu dùng LLM cho toàn bộ tin: 10 triệu tin × 100 token = 1 tỉ token input ≈ $100/ngày, cộng ~$5 output, tổng khoảng $105/ngày ≈ $3.150/tháng (theo giá 0,10 và 0,50 mỗi 1 triệu token). Nếu chỉ khoảng 10% tin qua LLM thì còn khoảng $315/tháng (10% là giả định, cần đo lại trên tập val).
Đánh đổi chấp nhận: tin lưỡng lự bị trả quyết định chậm hơn (vài giây), cần theo dõi tỉ lệ tin chuyển lên LLM, và vẫn cần bước lọc đầu vào ngoài LLM để chống prompt injection.

## 7. AI log và điều đã học

| Phần | Em hỏi AI gì | AI làm gì | Em sửa / kiểm chứng |
|---|---|---|---|
| 1 (tự code) | Dán code tự viết bị lỗi, hỏi cách sửa | Chỉ lỗi trong code của em (sai tên biến, `X.t`, nhầm FP/FN...) và đưa dòng sửa | Tự sửa, chạy ô kiểm tra, giải thích được lỗi |
| 2 (tự code) | Hỏi lỗi `nn.ReLu` và thiếu `SEED` | Chỉ ra viết sai tên lớp, chạy lại từ đầu | Tự sửa và chạy lại |
| 3 (tự code) | Hỏi cách tính cosine, top-k, nhờ xem `knn_predict` | Giải thích công thức, xem lại code của em | Tự viết, chạy ô kiểm tra, so kNN với LogReg |
| 4 (được dùng AI) | Xin code gọi LLM và hướng dẫn tạo key Groq | Viết 4 hàm Phần 4, hướng dẫn cấu hình | Đọc lại, bỏ docstring lỗi, thêm `sleep` tránh 429, chạy mock test; key chỉ để trong Secrets |
| 5 (được dùng AI) | Hỏi cách đọc bảng, Pareto, bộ test khó, soạn câu trả lời | Diễn giải kết quả, đưa bản nháp, từ chối làm hộ thử thách | Đối chiếu số với bảng, diễn đạt lại, tự làm tính tay |
| Thử thách | Nhờ làm hộ | Từ chối, nhắc cách làm và chỉ ra lỗi cú pháp, nhánh in kết quả bịa | Tự viết 4 tin; chưa chạy được thật (xem mục 5) |

**Điều em hiểu ra hôm nay:** F1 cao trên tập test thường không chứng minh mô hình hiểu spam, vì tập đó cùng phân phối với tập train. Chỉ khi thử trên bộ case khó em mới thấy BoW tụt từ ~0,89 xuống ~0,35 và embedding bắt chủ đề chứ chưa bắt ý định. Em cũng hiểu không có phương án "tốt nhất tuyệt đối" mà phải cân bằng giữa chất lượng, độ trễ và chi phí, và thiết kế lai (router) thường hợp lý hơn dùng một cách duy nhất.
