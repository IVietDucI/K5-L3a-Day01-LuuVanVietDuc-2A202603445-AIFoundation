# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **SEED:** 9276

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | có |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | có |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | có |

## 2. Kết quả chính

| Phương pháp | F1 test | F1 SUB (40) | Train (s) | Throughput (tin/s) | p95 (ms/tin) | USD/1.000 tin (đã đo) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BoW + Logistic Regression | 0.890 | 0.889 | 3.476 | 30,898.723 | 0.083 | 0.000 |
| BoW + MLP | 0.955 | 0.947 | 4.642 | 50,020.986 | 0.194 | 0.000 |
| MiniLM + kNN | 0.870 | 0.976 | 65.819 | 33.988 | 97.056 | 0.000 |
| MiniLM + Logistic Regression | 0.944 | 0.974 | 66.231 | 50.583 | 32.745 | 0.000 |
| MiniLM + MLP head | 0.957 | 1.000 | 67.068 | 50.407 | 31.769 | 0.000 |

Nhận xét: Trên test thông thường, MiniLM + MLP head có F1 cao nhất (0.957), nhưng BoW + MLP gần tương đương (0.955) và có độ trễ nhỏ hơn nhiều. MiniLM + kNN có F1 SUB khá cao (0.976) nhưng p95 = 97.056 ms, vượt ngưỡng 50 ms. Giá trị USD/1.000 tin bằng 0 ở các mô hình nội bộ chỉ biểu thị không có chi phí gọi API bên ngoài được đo; không có nghĩa là chạy hệ thống không tốn tài nguyên hoặc hạ tầng.

Pareto: Ba phương pháp trên đường Pareto theo 1 − F1_SUB và p95 là BoW + Logistic Regression, BoW + MLP và MiniLM + MLP head. Bốn phương pháp đạt p95 < 50 ms là hai mô hình BoW, MiniLM + Logistic Regression và MiniLM + MLP head.

## 3. Dự đoán và thực tế

(1) dl_curve — F1 của MLP so với Logistic Regression. Trước thí nghiệm, tôi dự đoán F1 của MLP sẽ xấp xỉ Logistic Regression và Validation Loss giảm ở những epoch đầu rồi tăng trở lại. Thực tế, F1 test của MLP là 0.955, cao hơn Logistic Regression (0.890) khoảng 0.065, nên phần chất lượng mô hình không đúng như tôi dự đoán. Tuy vậy, xu hướng overfitting có cơ sở: epoch có Validation Loss thấp nhất là 6. Điều này cho thấy lớp ẩn và ReLU có thể giúp mô hình biểu diễn dữ liệu tốt hơn, dù cần theo dõi overfitting.

(2) dl_relu — Bỏ ReLU. Tôi dự đoán việc bỏ ReLU sẽ làm F1 giảm về gần mức Logistic Regression (xấp xỉ 0.890). Tuy nhiên, output thí nghiệm ghi F1 test = 0.955 dù không dùng ReLU. Về lý thuyết, chuỗi lớp Linear không có ReLU có thể quy về một phép biến đổi tuyến tính; nhưng kết quả thử nghiệm còn phụ thuộc cách tối ưu, số epoch và bộ dữ liệu. Không thể kết luận chỉ từ một lần chạy rằng ReLU luôn làm F1 tăng.

(3) hard_eval — Độ bền trước dữ liệu khó. Trước thí nghiệm, tôi dự đoán MiniLM + MLP head đạt 70–85% accuracy trên 50 case khó. Kết quả thật chỉ 64% accuracy, với F1 = 0.654; ngược lại MiniLM + kNN đạt 76% accuracy, F1 = 0.786 và tốt nhất trong bộ test khó ban đầu. Dự đoán này lệch vì tôi đã đánh giá quá cao khả năng tổng quát của mô hình có F1 SUB = 1.000. Tập SUB chỉ có 40 tin nên dễ tạo ấn tượng lạc quan.

## 4. Phân tích 50 case khó

| Mô hình | F1 SUB 40 | F1 50 case khó | Mức giảm F1 | Accuracy 50 case |
| --- | ---: | ---: | ---: | ---: |
| BoW + Logistic Regression | 0.889 | 0.343 | 0.546 | 54% |
| BoW + MLP | 0.947 | 0.410 | 0.537 | 54% |
| MiniLM + kNN | 0.976 | 0.786 | 0.190 | 76% |
| MiniLM + Logistic Regression | 0.974 | 0.612 | 0.362 | 62% |
| MiniLM + MLP head | 1.000 | 0.654 | 0.346 | 64% |

BoW + Logistic Regression giảm F1 nhiều nhất: 0.546 điểm, từ 0.889 xuống 0.343. Nhóm khó nổi bật với BoW là spam giọng tự nhiên (BoW + LogReg đúng 10%, BoW + MLP đúng 10%) và né từ khóa (lần lượt đúng 17% và 0%). BoW phụ thuộc vào sự xuất hiện của từ riêng lẻ, vì vậy có thể bỏ sót ý định quảng cáo hoặc lừa đảo khi câu viết giống lời hỏi thăm hay ký tự bị thay đổi.

Hai ví dụ cụ thể từ bộ test:
- Case #12 — spam giọng tự nhiên: Tin bắt đầu bằng cách hỏi thăm lịch sự, sau đó hướng người nhận tới lời mời không mong muốn. Cả năm mô hình đều dự đoán sai; cách mở đầu giống giao tiếp cá nhân khiến mục đích spam khó nhận ra chỉ qua từ ngữ.
- Case #15 — spam giọng tự nhiên: Tin giới thiệu dịch vụ dọn dẹp và viện dẫn lời giới thiệu của hàng xóm để tạo sự tin cậy. Cả năm mô hình đều dự đoán sai; nội dung không có những dấu hiệu spam kinh điển như “win” hoặc “prize”.
Trên 50 case gốc, có 5 case mà cả năm mô hình cùng sai; tất cả thuộc nhóm spam giọng tự nhiên. Sau khi bổ sung 5 case Bonus do AI gợi ý, kết quả đánh giá 55 case trong notebook là F1 lần lượt 0.324 / 0.390 / 0.733 / 0.604 / 0.643 và accuracy 54.5% / 54.5% / 70.9% / 61.8% / 63.6% (theo thứ tự các mô hình ở bảng trên). Khi đó có 6 case mà tất cả mô hình cùng sai, trong đó có case Bonus #100. Hai mốc 50 và 55 case được báo cáo riêng, không trộn lẫn.

Bộ stress test có ích để phát hiện điểm yếu và đánh giá khả năng chịu dữ liệu lệch phân phối, nhưng vì được thiết kế thủ công và quy mô nhỏ, không thể coi điểm trên đó là ước lượng đại diện cho toàn bộ SMS thực tế. Phần thử nghiệm LLM và hiệu quả phòng thủ prompt injection chưa có số liệu vì chưa gọi được API thật.

## 5. Thử thách cá nhân

SEED: 9276 — Đề #1: Thử ngưỡng phân loại 0.2, 0.5 và 0.8 với mô hình ML Phần 1; so sánh Precision, Recall, F1 và giải thích lựa chọn ngưỡng cho ngân hàng cần tránh chặn nhầm OTP.
Trạng thái: Chưa thực hiện phần code và chưa có số liệu Precision, Recall, F1 cho ba ngưỡng. Vì vậy báo cáo không đưa ra kết quả giả định.
Hiểu biết lý thuyết: Tăng ngưỡng từ 0.5 lên 0.8 thường giúp giảm số tin ham bị đánh dấu spam (False Positive), phù hợp hơn với yêu cầu hạn chế chặn nhầm OTP, nhưng có thể làm tăng số spam lọt qua. Đây mới là nhận định lý thuyết, chưa phải kết luận thực nghiệm của thử thách.

## 6. Phương án cho sản phẩm thật

Nhà mạng cần xử lý 10 triệu SMS/ngày, tương đương khoảng 116 SMS/giây trung bình, với SLA p95 < 50 ms/tin. Từ thí nghiệm, bốn mô hình đạt yêu cầu p95, nhưng throughput của các phương pháp MiniLM chỉ khoảng 50–51 tin/giây trong phép đo hiện tại, thấp hơn tải trung bình yêu cầu nếu chạy trên một cấu hình giống như thí nghiệm. Cần tối ưu hoặc mở rộng hạ tầng để đáp ứng lưu lượng thực tế và các giờ cao điểm.

Tôi chọn BoW + MLP làm tuyến xử lý nhanh ban đầu: F1 test = 0.955, p95 = 0.194 ms, 50,020.986 tin/giây trong phép đo inference cả lô. Đổi lại, BoW + MLP chỉ đạt F1 = 0.410 trên 50 case khó, đặc biệt yếu trước spam giọng tự nhiên và né từ khóa. Vì vậy không nên dùng riêng mô hình này để đưa ra mọi quyết định chặn tự động.

Thiết kế lai (router): Một mô hình BoW/Logistic Regression tính xác suất spam trước; tin có điểm rất thấp hoặc rất cao được xử lý tại tuyến nhanh, còn tin lưỡng lự (ví dụ p trong khoảng 0.3–0.7) chuyển sang MiniLM + MLP head. Trường hợp rủi ro cao hoặc hai kết quả không thống nhất có thể chuyển sang bước kiểm tra bổ sung. Cách này tiết kiệm tài nguyên so với chạy encoder cho tất cả SMS nhưng vẫn có khả năng phân tích ngữ nghĩa ở các ca khó. Phải kiểm thử bổ sung tỷ lệ định tuyến, tính hiệu chuẩn xác suất, độ trễ end-to-end và thông lượng trước khi triển khai thực tế.

Chi phí LLM tham khảo: Với giả định của đề bài 100 input tokens + 1 output token/SMS, giá 0.10 USD/triệu input tokens và 0.50 USD/triệu output tokens, nếu dùng LLM cho toàn bộ 10 triệu SMS/ngày thì chi phí token khoảng 105 USD/ngày, hay 3.150 USD/30 ngày. Đây chỉ là phép tính theo giả định, chưa gồm hạ tầng, lưu lượng token thực tế và chưa có benchmark LLM do không có API khả dụng.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| Phần 0 — Dữ liệu | Giải thích data leakage, bỏ trùng và tách train/val/test. | AI giải thích nguy cơ kết quả đánh giá bị lạc quan khi dữ liệu trùng hoặc gần trùng. | Tôi kiểm tra tập dữ liệu sau bỏ trùng và cách chia train/val/test. |
| Phần 1 — ML | Hỏi về BoW, Sigmoid, Gradient Descent, learning rate, Precision/Recall/F1. | AI giải thích công thức và cách đọc kết quả; hỗ trợ phân tích dự đoán. | Tôi đối chiếu kết quả F1, các biểu đồ Loss và những tin bị dự đoán sai. |
| Phần 2 — DL | Hỏi về MLP, ReLU, Backpropagation, overfitting và xu hướng Loss. | AI giải thích cơ chế huấn luyện và hỗ trợ diễn giải thí nghiệm. | Tôi kiểm tra biểu đồ, epoch val loss tối ưu và kết quả thử bỏ ReLU. |
| Phần 3 — Embedding | Hỏi về Sentence Embedding, Cosine Similarity, kNN và tìm kiếm theo ngữ nghĩa. | AI giải thích khác biệt giữa BoW và biểu diễn ngữ nghĩa, hỗ trợ phân tích kết quả. | Tôi kiểm tra Top-k và so sánh BoW đúng 4/6, Encoder + LogReg đúng 5/6 trên bộ tricky. |
| Phần 4 — API Configuration | Hỏi về API key an toàn và lỗi 401. | AI hướng dẫn `getpass`, cách hiểu lỗi xác thực và xử lý khi không thể gọi API. | Tôi chạy bài kiểm tra giả lập, đặt `HAS_LLM = False` khi không có key hợp lệ. |
| Phần 4 — TODO 4.1–4.4 | Hỏi cách triển khai `call_openai`, tạo prompt, parser nhãn và phân loại batch. | AI cung cấp mã cho các hàm LLM và giải thích đo latency/ước tính chi phí. | Tôi chạy các check giả lập, không ghi nhận F1/chi phí thực tế của LLM. |
| Phần 4 — Dự đoán, bắt lỗi | Hỏi về Zero-shot, Few-shot, temperature, max_tokens và cách parse nhãn. | AI gợi ý giả thuyết và giải thích lỗi phân loại chuỗi. | Tôi ghi các dự đoán `llm_models`, `llm_fewshot`; chưa có phép đo API. |
| Phần 5 — Pareto | Hỏi cách phân tích F1, p95 và đường Pareto. | AI hỗ trợ giải thích các mô hình bị lấn át và SLA. | Tôi chạy biểu đồ, đối chiếu ba mô hình Pareto và bốn mô hình dưới SLA. |
| Phần 5 — Stress Test | Hỏi vì sao điểm F1 giảm trên 50 case khó và nguyên nhân mô hình sai. | AI hỗ trợ diễn giải kết quả từng nhóm và các case cùng bị phân loại sai. | Tôi đối chiếu 50 case gốc và 55 case sau Bonus; không coi LLM là đã được đánh giá. |
| Bonus — MY_CASES | Nhờ AI đề xuất SMS gây nhầm lẫn và `why_tricky`. | AI gợi ý 5 case, nhãn và lý do gây khó. | Tôi chạy lại với 55 case, nhưng các case ban đầu có nguồn gốc từ AI, chưa phải 5 case hoàn toàn tự nghĩ. |
| Phần 5 — Tính tay | Hỏi cách tính baseline, sigmoid, F1, cosine và chi phí LLM. | AI giải thích công thức và gợi ý các bước tính. | Tôi xác nhận 5 kết quả hash; ảnh trình bày tay vẫn cần tự hoàn thiện. |
| Thử thách cá nhân | Hỏi khái niệm thay đổi threshold, False Positive và Recall. | AI giải thích nguyên lý và có cung cấp code minh họa. | Tôi chưa hoàn thành thí nghiệm cá nhân và không ghi số liệu không có. |
| Phần 5 — Phản ánh | Hỏi cách đánh giá mô hình cho 10 triệu SMS/ngày và xây dựng Hybrid Router. | AI hỗ trợ bản nháp lập luận và tính chi phí theo giả định. | Tôi đối chiếu với F1, p95 và throughput đo được; ghi rõ các hạn chế chưa kiểm chứng. |

Điều đã học: Trước bài lab, tôi dễ đánh giá mô hình bằng một giá trị Accuracy/F1 duy nhất; sau khi làm, tôi hiểu rằng phải xem cả Precision, Recall, dữ liệu lệch lớp và chi phí dự đoán sai. Tôi cũng thấy F1 cao trên một tập SUB nhỏ không đảm bảo mô hình bền vững trước tin nhắn khó. Khi chọn mô hình triển khai, phải xét đồng thời chất lượng, độ trễ, thông lượng, tính ổn định và yêu cầu thực tế của hệ thống.
