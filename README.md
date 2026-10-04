# Phân tích giao dịch bán lẻ điện tử

Repo này phân tích file giao dịch bán lẻ trong [main.ipynb](main.ipynb) và đề xuất các bài toán có thể làm tiếp. Dữ liệu nguồn đặt tại `data/electronics-retail-transaction.xlsx`; thư mục `data/` được loại khỏi Git.

## Dữ liệu và phạm vi

- 11.800 dòng giao dịch, 6.453 đơn, 11 cửa hàng, 2 mã ca, 1.507 SKU.
- Thời gian sau khi sửa cách đọc ngày Excel: **01/08–09/09/2026**, 40 ngày liên tục.
- Mỗi dòng là một mặt hàng hoặc dịch vụ trong đơn. Khóa đơn thực dụng là `Stt đơn hàng`, không phải `ID`.
- Không có customer ID đáng tin, tồn kho, lịch nhân sự, giá vốn hay nhật ký khách được chào sản phẩm. Các giới hạn này quyết định bài toán nào có thể đánh giá.

Các số dưới đây được tính trên file hiện có. **Thử nghiệm được** nghĩa là có thể định nghĩa nhãn và đo sai số ban đầu; chưa có nghĩa là nên triển khai tự động.

## 1. Dự báo tải giao dịch theo cửa hàng và ca — thử nghiệm được

**Ai dùng, để làm gì:** trưởng cửa hàng và vận hành xem ca nào ngày hôm sau có thể đông đơn để điều chỉnh phân công người hoặc quầy phục vụ.

- **Đơn vị:** cửa hàng × mã ca × ngày lịch. **Đầu ra:** số đơn bán lẻ khác nhau trong ca đó.
- **Đầu vào có trước:** cửa hàng, mã ca, thứ trong tuần và số đơn các ngày trước. Không dùng dữ liệu của ngày cần dự báo.
- **Cơ sở dữ liệu:** 845/880 ô cửa hàng–ca–ngày có đơn bán lẻ. 35 ô còn lại chưa rõ là ca nghỉ, không bán hay thiếu dữ liệu.
- **Kiểm chứng ban đầu:** giữ 03/09–09/09 làm tập cuối; 118/154 ô đủ lịch sử cho hai mốc so sánh. Dự báo bằng trung bình 7 ngày trước có MAE **3,21 đơn/ca**; đây chỉ là mốc đơn giản.
- **Cần thêm để dùng thật:** ý nghĩa mã ca, lịch nhân viên, giờ làm, thời gian phục vụ và nhiều tháng dữ liệu. Cần đo tác động lên ca thiếu/dư người, không chỉ MAE.

## 2. Dự báo lượng bán điện thoại và phụ kiện — thử nghiệm được

**Ai dùng, để làm gì:** quản lý danh mục theo dõi cửa hàng bán nhanh để ưu tiên kiểm tra phương án phân bổ hàng. Dữ liệu hiện có chưa đủ để tự quyết định số lượng nhập/chuyển.

- **Đơn vị:** cửa hàng × chủng loại (`DIENTHOAI` hoặc `PHUKIEN`) × ngày. **Đầu ra:** tổng `Số lượng` ở dòng bán lẻ có doanh thu và số lượng dương.
- **Đầu vào có trước:** cửa hàng, chủng loại, thứ trong tuần và lượng bán các ngày trước.
- **Cơ sở dữ liệu:** 868/880 ô cửa hàng–nhóm–ngày có bán. Chọn hai nhóm này vì độ phủ cao; 1.507 SKU riêng lẻ quá thưa để áp cùng bài toán.
- **Kiểm chứng ban đầu:** 147/154 ô cuối kỳ đủ lịch sử. MAE của mốc trung bình 7 ngày là **2,79 điện thoại** và **4,67 phụ kiện** mỗi cửa hàng–ngày.
- **Giới hạn:** số *đã bán* có thể thấp hơn nhu cầu khi hết hàng. Muốn hỗ trợ quyết định nhập/chuyển cần tồn kho, hàng đang về, thời gian cung ứng và thêm dữ liệu lịch sử.

## 3. Phân cụm giỏ hàng theo kiểu mua — khám phá

**Ai dùng, để làm gì:** bộ phận danh mục và trưng bày tìm các kiểu lượt mua để thiết kế kệ hoặc kịch bản tư vấn đem đi thử.

- **Đơn vị:** `Stt đơn hàng`, tức một lượt mua, **không phải khách hàng**. Có 6.262 giỏ bán lẻ có doanh thu dương; 2.219 giỏ (35,4%) có ít nhất hai chủng loại.
- **Đặc trưng ứng viên:** nhóm hàng có/không trong giỏ, số nhóm, dải giá trị đơn; không dùng mã đơn hay tên khách làm tín hiệu.
- **Cách làm:** so với phân nhóm bằng quy tắc dễ hiểu trước; chỉ dùng thuật toán phân cụm nếu nhóm ổn định giữa các tuần, dễ giải thích và dẫn đến hành động khác nhau. Đánh giá tác động trưng bày bằng thử nghiệm tại cửa hàng.
- **Giới hạn:** 4.043 giỏ chỉ có một chủng loại; dữ liệu không ghi trình tự thêm hàng vào giỏ. Không có cơ sở để gọi cụm giỏ là phân khúc khách hoặc xây bước dự đoán kiểu giỏ trước mua.

## 4. Tìm cặp nhóm hàng để thử bán kèm — khám phá

**Ai dùng, để làm gì:** bộ phận danh mục lập danh sách cặp hàng để thử đặt gần nhau hoặc gợi ý tại quầy.

- **Đơn vị:** giỏ bán lẻ có doanh thu dương. Đánh giá bằng số đơn cùng xuất hiện, support, confidence, lift; lọc cặp ít đơn và kiểm tra độ ổn định theo tuần/cửa hàng.
- **Ví dụ trong file:** điện thoại–phụ kiện có **1.276 đơn**, lift **0,902**; điện thoại–HCare có **775 đơn**, lift **1,505**. Cặp phổ biến chưa chắc có lift cao.
- **Giới hạn:** đồng mua không chứng minh việc gợi ý sẽ làm tăng doanh số. Riêng HCare còn thiếu dữ liệu khách có được chào gói và thiết bị có đủ điều kiện. Cần thử nghiệm có nhóm đối chứng trước khi khẳng định hiệu quả.

## 5. Ưu tiên đối soát đơn thiếu số hóa đơn — quy tắc nghiệp vụ

**Ai dùng, để làm gì:** kế toán và vận hành lập danh sách đơn cần rà soát theo loại bán, cửa hàng và ngày.

- File có **124 đơn** thiếu số hóa đơn, trong đó **103 đơn bán dịch vụ** và **5 đơn bán lẻ**. Thiếu số có thể phù hợp với quy trình của từng loại giao dịch.
- Bắt đầu bằng quy tắc và đối chiếu chứng từ; chưa cần mô hình phát hiện gian lận. Cần xác nhận loại giao dịch nào bắt buộc có hóa đơn trước khi gắn cờ lỗi.

## Những bài toán chưa đủ dữ liệu

- **Phân cụm khách hàng → dự đoán quay lại/CLV:** chỉ có tên, không có định danh khách ổn định; 40 ngày quá ngắn cho vòng đời.
- **Dự báo tồn kho hoặc nhu cầu từng SKU:** thiếu tồn kho và dấu hiệu hết hàng; SKU trung vị chỉ xuất hiện trong 2 đơn có doanh thu dương.
- **Dự đoán khách mua HCare trước tư vấn:** thiếu nhật ký gói đã được chào và điều kiện áp dụng; nhãn mua kèm có thể phản ánh quy trình bán hiện tại.

## Chạy notebook

Đặt file nguồn vào `data/electronics-retail-transaction.xlsx`, cài `pandas`, `numpy`, `openpyxl` và Jupyter, rồi chạy [main.ipynb](main.ipynb) từ trên xuống. Notebook giữ phần EDA và các ô tạo số liệu dùng trong README; file Excel không được đưa vào Git.

## Tài liệu tham khảo

- [Retail business analytics: Customer visit segmentation using market basket data](https://www.sciencedirect.com/science/article/pii/S0957417418300356) — phân nhóm lượt mua từ giỏ hàng, không cần lịch sử định danh khách.
- [Retail store scheduling for profit](https://www.sciencedirect.com/science/article/pii/S0377221714004561) — kết hợp dự báo với lịch nhân sự bán lẻ.
- [Forecasting: Principles and Practice — Evaluating point forecast accuracy](https://otexts.com/fpp3/accuracy.html) — giữ dữ liệu cuối kỳ để đánh giá dự báo.
- [FreshRetailNet-50K](https://arxiv.org/abs/2505.16319) — minh họa vấn đề doanh số bị giới hạn bởi tình trạng hết hàng.
