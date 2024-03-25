# PhanLoaiUngThuVu_BreastCancerClassificationKNN_MachineLearning
# Phân Loại Ung Thư Vú Bằng Cách Sử Dụng - KNN 

TÓM TẮT

Chương 1: Mở Đầu

•	Chương này giới thiệu về bối cảnh nghiên cứu, mục tiêu và ý nghĩa của việc áp dụng mô hình K─NN trong phân loại ung thư vú.

• Đặc biệt, nêu rõ vấn đề mất cân bằng trong bộ dữ liệu Breast Cancer và tầm quan trọng của quá trình tinh chỉnh siêu tham số.

Chương 2: Cơ Sở Lý Thuyết

•	Chương này trình bày cơ sở lý thuyết liên quan đến mô hình K─NN, bao gồm nguyên lý hoạt động, ưu điểm và nhược điểm của mô hình.

•	Giải thích cách mất cân bằng trong dữ liệu ảnh hưởng đến mô hình và cách phân chia dữ liệu phân tầng giúp giải quyết vấn đề này.

Chương 3: Phương Pháp Nghiên Cứu

•	Chương này tập trung vào quá trình chuẩn bị dữ liệu, lựa chọn đặc trưng và khám phá dữ liệu trước khi áp dụng mô hình.

•	Thực hiện các thử nghiệm với kích thước dữ liệu khác nhau để đánh giá ảnh hưởng lên hiệu suất của mô hình.

Chương 4:  Quy Trình Đào Tạo Và Đánh Giá Mô Hình

•	Chương này bao gồm quá trình triển khai và tinh chỉnh mô hình K─NN.

•	Sử dụng phương pháp phân tầng trong việc chia dữ liệu huấn luyện và kiểm tra để giải quyết vấn đề mất cân bằng.

•	Thực hiện tinh chỉnh siêu tham số bằng cách sử dụng cả Phương pháp Elbow và Tìm kiếm lưới để đạt được hiệu suất tối ưu.

•	Cung cấp kết quả chi tiết về độ chính xác và các thước đo đánh giá khác của mô hình sau quá trình tinh chỉnh.

Chương 5: Triển Khai Ứng Dụng & Phương Hướng Phát Triển

•	Chương cuối cùng đưa ra quá trình triển khai ứng dụng và đề xuất hướng phát triển tương lai.

•	 Nó cung cấp một tóm tắt về cài đặt và sử dụng, đồng thời đặt ra những thách thức hiện tại và đề xuất các cải tiến như chi tiết hóa thông tin, nhận biết trường 
hợp không mắc bệnh, và tối ưu hóa giao diện người dùng.

-----------------------------------------------------------------------

TỔNG QUAN:

■ Các Công Trình Nghiên Cứu Trong Nước:

•	Phân Loại Hình Ảnh Y Tế: Nhiều nghiên cứu tập trung vào phân loại hình ảnh y tế sử dụng học máy để chẩn đoán ung thư vú. Tuy nhiên, một số hạn chế về chi tiết và khả năng chẩn đoán chưa đạt đến mức độ mong đợi.

•	Khảo Sát Dữ Liệu Lâm Sàng: Một số nghiên cứu đã khảo sát các dữ liệu lâm sàng để đánh giá tình trạng sức khỏe của bệnh nhân. Tuy nhiên, nhu cầu chi tiết hóa thông tin về giai đoạn và nhận biết trường hợp không mắc bệnh chưa được đáp ứng đầy đủ.

■ Các Công Trình Nghiên Cứu Ngoại Nước:

•	Sự Kết Hợp Các Phương Pháp Học Máy: Nghiên cứu quốc tế thường kết hợp nhiều phương pháp học máy để tăng cường độ chính xác của chẩn đoán. Tuy nhiên, một số nghiên cứu còn đang đối mặt với thách thức trong việc tích hợp thông tin chi tiết.

•	Giao Diện Người Dùng Thân Thiện: Một số ứng dụng chẩn đoán đã tập trung vào việc xây dựng giao diện người dùng thân thiện để tạo trải nghiệm tích cực cho người dùng. Tuy nhiên, còn khả năng cải thiện để làm cho ứng dụng dễ sử dụng hơn.

■ Vấn Đề Còn Tồn Tại và Nhu Cầu Nghiên Cứu:

Thiếu Chi Tiết Về Giai Đoạn Bệnh:

•	Nhiều ứng dụng chỉ chẩn đoán lành tính hoặc ác tính mà không cung cấp thông tin chi tiết về giai đoạn của ung thư vú.

Khả Năng Nhận Biết Trường Hợp Không Mắc Bệnh:

•	Cần có khả năng nhận biết và loại bỏ trường hợp người hoàn toàn không mắc bệnh để tăng độ tin cậy của ứng dụng.

Giao Diện Người Dùng Cần Cải Thiện:

•	Tính thân thiện và tính ứng dụng của giao diện người dùng có thể được cải thiện để tối ưu hóa trải nghiệm người dùng.

Kết Luận:

Tổng quan đã làm rõ rằng mặc dù đã có những tiến bộ trong lĩnh vực chẩn đoán ung thư vú sử dụng học máy, nhưng vẫn còn những thách thức cần vượt qua. Đề tài này đặt ra nhiệm vụ tập trung vào cải thiện chi tiết thông tin để làm cho ứng dụng trở nên thực sự hữu ích trong lĩnh vực y tế.

 
CHƯƠNG 1:  MỞ ĐẦU

1.1.	Lý do chọn đề tài:

Việc chọn đề tài về phân loại ung thư vú bằng mô hình K─NN (K─Nearest Neighbors) đặt ra từ nhu cầu cấp bách trong lĩnh vực y học. Ung thư vú là một trong những căn bệnh phổ biến nhất và có tác động sâu rộng đến sức khỏe và tâm lý của phụ nữ trên toàn cầu. Để giải quyết thách thức này, việc áp dụng công nghệ thông tin, đặc biệt là học máy, trở nên cực kỳ quan trọng để tạo ra các công cụ hỗ trợ chẩn đoán hiệu quả và chính xác.

1.1.1. Mô tả tình trạng nguy cơ và tần suất của bệnh ung thư vú:

Bệnh ung thư vú là một trong những nguyên nhân hàng đầu gây tử vong do bệnh lý ung thư ở phụ nữ trên khắp thế giới. Tình trạng nguy cơ mắc bệnh ung thư vú không chỉ liên quan đến yếu tố gen di truyền mà còn ảnh hưởng bởi nhiều yếu tố môi trường, lối sống và các yếu tố khác. Tần suất mắc bệnh ung thư vú ngày càng tăng, đặc biệt là ở những quốc gia đang phát triển, làm tăng áp lực đối với hệ thống y tế và nhu cầu về các phương pháp dự đoán và phòng ngừa.

1.1.2. Ý nghĩa và quan trọng của việc sử dụng machine learning để dự đoán bệnh ung thư vú:

Việc sử dụng machine learning (học máy) trong dự đoán bệnh ung thư vú mang lại nhiều ưu điểm lớn. Mô hình machine learning có khả năng xử lý lượng lớn dữ liệu và phát hiện các mối liên quan phức tạp giữa các yếu tố, điều này giúp nâng cao khả năng dự đoán và chẩn đoán bệnh. Đồng thời, nó cũng có thể hỗ trợ trong việc tìm ra các đặc điểm chính và mô hình hóa nguyên nhân của bệnh, từ đó giúp cải thiện quá trình chẩn đoán sớm và đặt ra các biện pháp phòng ngừa hiệu quả.

Sự kết hợp giữa kiến thức y học và khả năng dự đoán của machine learning tạo ra cơ hội để phát triển các mô hình dự đoán chính xác, có thể hỗ trợ các chuyên gia y tế trong việc đưa ra quyết định và tư vấn điều trị. Điều này không chỉ giảm áp lực cho hệ thống y tế mà còn cung cấp lợi ích to lớn cho cộng đồng, đặc biệt là trong việc tăng cường khả năng dự đoán và chẩn đoán sớm bệnh ung thư vú, từ đó cải thiện khả năng chữa trị và tăng cơ hội sống sót của bệnh nhân.

1.2 Mục tiêu nghiên cứu:
Mục tiêu của nghiên cứu này là xây dựng một mô hình phân loại ung thư vú dựa trên dữ liệu y khoa sử dụng thuật toán K─NN. Tiến hành thu thập dữ liệu từ các nguồn tin cậy như cơ sở dữ liệu ung thư vú và các chỉ số y khoa khác. Mục tiêu chính là huấn luyện một mô hình K─NN đáng tin cậy, có khả năng phân loại các trường hợp ung thư vú thành các nhóm có tính chất lành tính và ác tính.

1.2.1 Hiểu rõ các yếu tố quyết định:

 Phân tích mối quan hệ giữa các yếu tố quyết định và kết quả dự đoán từ mô hình K─NN, giúp tăng cường sự hiểu biết về các đặc điểm quan trọng liên quan đến bệnh ung thư vú.

1.2.2 Kết quả dự kiến:

Một mô hình K─NN hiệu quả trong việc dự đoán bệnh ung thư vú, có khả năng đánh giá và so sánh với các phương pháp khác. Kết quả nghiên cứu dự kiến sẽ cung cấp thông tin hữu ích và hỗ trợ quyết định cho lĩnh vực y tế trong việc đối phó với bệnh ung thư vú, đặc biệt là trong việc phòng ngừa và chẩn đoán sớm.

1.3 Phạm vi và giới hạn của đề tài:

Nghiên cứu sẽ tập trung vào việc xây dựng và kiểm định mô hình phân loại bằng K─NN trên một tập dữ liệu cụ thể, có thể bao gồm các thông tin như kích thước khối 
u, các đặc điểm hoặc các chỉ số y khoa quan trọng khác. Tuy nhiên, sẽ không đi sâu vào các phương pháp chẩn đoán hay điều trị cụ thể, chỉ tập trung vào quá trình phân loại dựa trên dữ liệu có sẵn.

Điều quan trọng cần lưu ý là mặc dù K─NN là một thuật toán mạnh mẽ, việc áp dụng nó vào thực tế y học đòi hỏi sự thận trọng và kiểm tra kỹ lưỡng để đảm bảo tính chính xác và an toàn trong việc đưa ra quyết định lâm sàng.

CHƯƠNG 2: CƠ SỞ LÝ THUYẾT

2.1  Đặc điểm, nguyên nhân và tiến triển bệnh ung thư vú:

Đặc Điểm:

  Ung thư vú là một loại ung thư phổ biến ở phụ nữ, xuất phát từ tế bào vú. Bệnh này thường phát triển một cách không đau đớn và khó nhận biết ở các giai đoạn đầu. Các đặc điểm chính bao gồm sự xuất hiện của khối u, thay đổi hình dạng vú, và có thể đi kèm với các triệu chứng như đau ngực.

Nguyên Nhân và Tiến Triển:
    
  Nguyên nhân chính của ung thư vú có thể liên quan đến yếu tố gen, môi trường, và lối sống. Quá trình tiến triển của bệnh thường đi qua các giai đoạn từ việc tế bào bình thường biến đổi thành tế bào ung thư, và từ đó lan ra các cấp độ nghiêm trọng khác nhau.

2.2 Thuật toán K-Nearest Neighors (K─NN):

2.2.1 Nguyên Tắc Hoạt Động:

Thuật toán K─NN là một phương pháp học máy dựa trên việc đo lường sự tương đồng giữa các điểm dữ liệu. Nguyên lý hoạt động của K─NN là dự đoán nhãn của một điểm dữ liệu mới bằng cách xem xét những điểm dữ liệu gần nhất trong không gian đặc trưng và gán nhãn dự đoán dựa trên đa số những điểm gần nhất đó.

2.2.2 Ưu Điểm và Nhược Điểm:

Ưu Điểm:

  Đơn giản và dễ triển khai: K─NN không yêu cầu giả định phức tạp và dễ hiểu, làm cho nó phù hợp cho nhiều bài toán.
  Không yêu cầu giả định về phân phối: Không đặt ra giả định về phân phối của dữ liệu, điều này làm cho K─NN linh hoạt với nhiều loại dữ liệu.

Nhược Điểm:
  
  Nhạy cảm với nhiễu: K─NN có thể bị ảnh hưởng bởi các nhiễu và điểm dữ liệu cách xa nhau trong không gian đặc trưng.
  
  Yêu cầu nhiều tài nguyên: Đặc biệt là khi có số lượng lớn điểm dữ liệu, việc tính toán khoảng cách giữa chúng trở nên tốn kém.

2.2.3 Ứng dụng trong dự đoán bệnh lý:

Trong dự án này, chúng em sử dụng dữ liệu từ các bệnh nhân ung thư vú được nhập vào hệ thống từ tệp CSV (Comma─Separated Values). Đây là một tập hợp các thông tin liên quan đến bệnh nhân, bao gồm các yếu tố như kích thước u, đặc điểm của u, và kết quả các xét nghiệm y khoa.

Thuật toán K─Nearest Neighbors (K─NN) được triển khai để dự đoán khả năng mắc bệnh ung thư vú của một bệnh nhân dựa trên các đặc trưng này. Cụ thể, thông tin từ người bệnh được nhập vào hệ thống, ví dụ như kích thước u, đặc điểm của u và kết quả các xét nghiệm, được sử dụng làm đặc trưng để đưa ra quyết định.

Ví dụ, khi một người bệnh mới nhập thông tin của mình, hệ thống sẽ sử dụng K─NN để so sánh thông tin này với các bệnh nhân khác trong tập dữ liệu và dự đoán liệu người đó có khả năng mắc bệnh ung thư vú lành tính hay ác tính.

Thông qua quá trình nhập thông tin và ứng dụng của K─NN trong dự đoán bệnh lý, chúng em hy vọng mô hình có thể cung cấp thông tin hữu ích và chính xác về khả năng mắc bệnh ung thư vú lành tính ─ ác tính, giúp bác sĩ và chuyên gia y tế có thêm hỗ trợ trong quá trình chẩn đoán và đưa ra quyết định điều trị. Đồng thời, việc này có thể giúp giảm thiểu việc chẩn đoán trễ và tăng cường khả năng nắm bắt bệnh lý ở giai đoạn sớm.

CHƯƠNG 3: PHƯƠNG PHÁP NGHIÊN CỨU

3.1.Khám Phá Dữ Liệu:

3.1.1.Các Đặc Điểm trong Tập Dữ Liệu:

Thông số cơ bản:

─ Kiểu số thực (float) gồm:  31 thuộc tính 

─ Kiểu số nguyên (int) gồm:  1 thuộc tính

─ Kiểu đối tượng (object) gồm: 1 thuộc tính 

─ Số Trường Hợp: 569 trường hợp

─ Số Thuộc Tính: 33 thuộc tính

Phân Loại:

Bộ dữ liệu được phân loại thành hai nhóm chính:

─ Benign (Lành tính): Ung thư vú không nguy hiểm.

─ Malignant (Ác tính): Ung thư vú có nguy cơ cao và nguy hiểm hơn.

Định lượng:

 ─ Bao gồm các thuộc tính như : 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'.

Thông tin các đặc trưng : 

+ ID: (Số ID)

+ Diagnosis (Chẩn đoán): Chẩn đoán về mô mầm vú (M = ác tính, B = lành tính)

+ Radius_mean (Bán kính trung bình): Trung bình khoảng cách từ trung tâm đến các điểm trên đường viền

+ Texture_mean (Độ nhám trung bình): Độ lệch chuẩn của các giá trị mức xám

+ Perimeter_mean (Chu vi trung bình): Kích thước trung bình của khối u căn bản

+ Area_mean (Diện tích trung bình): Tính giá trị trung bình của diện tích của tất cả các khối u trong tập dữ liệu

+ Smoothness_mean (Độ mịn trung bình): Trung bình độ biến thiên cục bộ trong độ dài bán kính

+ Compactness_mean (Độ nén trung bình): Trung bình của chu vi bình phương / diện tích - 1.0

+ Concavity_mean (Độ lõm trung bình): Trung bình mức độ lõm của phần cung của đường viền

+ Concave points_mean (Số lõm trung bình): Trung bình số phần lõm của đường viền

+ Symmetry_mean (Đối xứng trung bình): Trung bình đối xứng của các điểm trong hình ảnh

+ Fractal_dimension_mean (Kích thước fractal trung bình): Trung bình của chiều sâu đối với một đối tượng fractal

+ Radius_se (Độ lệch chuẩn bán kính): Độ biến đổi của bán kính

+ Texture_se (Độ nhám lệch chuẩn): Độ biến đổi của độ nhám

+ Perimeter_se (Độ biến đổi chu vi): Độ biến đổi của chu vi

+ Area_se (Diện tích lệch chuẩn): Độ biến đổi của diện tích

+ Smoothness_se (Độ mịn lệch chuẩn): Độ biến đổi của độ mịn

+ Compactness_se (Độ nén lệch chuẩn): Độ biến đổi của độ nén

+ Concavity_se (Độ lõm lệch chuẩn): Độ biến đổi của độ lõm

+ Concave points_se (Số lõm lệch chuẩn): Độ biến đổi của số lõm

+ Symmetry_se (Đối xứng lệch chuẩn): Độ biến đổi của đối xứng

+ Fractal_dimension_se (Kích thước fractal lệch chuẩn): Độ biến đổi của kích thước fractal

+ Radius_worst (Bán kính tệ nhất): Giá trị lớn nhất của bán kính

+ Texture_worst (Độ nhám tệ nhất): Giá trị lớn nhất của độ nhám

+ Perimeter_worst (Chu vi tệ nhất): Giá trị lớn nhất của chu vi

+ Area_worst (Diện tích tệ nhất): Giá trị lớn nhất của diện tích

+ Smoothness_worst (Độ mịn tệ nhất): Giá trị lớn nhất của độ mịn

+ Compactness_worst (Độ nén tệ nhất): Giá trị lớn nhất của độ nén

+ Concavity_worst (Độ lõm tệ nhất): Giá trị lớn nhất của độ lõm

+ Concave points_worst (Số lõm tệ nhất): Giá trị lớn nhất của số lõm

+ Symmetry_worst (Đối xứng tệ nhất): Giá trị lớn nhất của đối xứng

+  Fractal_dimension_worst (Kích thước fractal tệ nhất): Giá trị lớn nhất của kích thước fractal

3.1.2. Loại bỏ dữ liệu:
 
 Các cột sau cần xem xét:

•	id: không thể được sử dụng cho mục đích phân loại

•	diagnosis ─ class labels (Nhãn ─labels): là một loại dữ liệu phân loại vì nó chứa các giá trị B ─ Lành tính và M ─ Ác tính.

•	Unnamed 32: Bao gồm giá trị NaN, do đó không hữu ích cho việc phân loại

Vì vậy, loại bỏ các cột trên khỏi tập dữ liệu.

_______________________________________________________________________
- Nhãn (labels) là cột "diagnosis"

labels_data = breast_cancer_data['diagnosis']
- Danh sách cột cần loại bỏ

danh_sach_loai_bo = ['Unnamed: 32', 'id', 'diagnosis']

- Các đặc điểm (features) là các cột còn lại

features_data = breast_cancer_data.drop(danh_sach_loai_bo, axis=1)

features_data.head(3)
________________________________________________________________________

3.2. Trực quan hóa dữ liệu & lựa chọn đặc điểm:

___________________________________________________________________________________________________________
- Hiển thị số lượng "Lành Tính " và "Ác Tính"
  
benign_count = labels_data.value_counts()['B']

malignant_count = labels_data.value_counts()['M']

print("Số lượng Lành Tính: {benign_count}\nSố lượng Ác tính: {malignant_count}"
.format(benign_count = labels_data.value_counts()[0], malignant_count = labels_data.value_counts()[1]))

plt.figure(figsize=(8,6))

labels_count = sns.countplot(x=labels_data, order=['M', 'B'])

plt.xlabel("Loại Ung Thư")

plt.ylabel("Số Lượng")

______________________________________________________________________________________________________________

 
Hình 3.2. 1: Hiển thị số lượng “Lành Tính” và “Ác Tính”

 Với biểu đồ trên: 
 
Số lượng Lành Tính: 357

Số lượng Ác tính: 212

 Có thể thấy tập dữ liệu không cân bằng, vì số mẫu lành tính lớn hơn số mẫu ác tính, vì vậy cần cân bằng tập dữ liệu.

3.2.1	Chuẩn hóa (Normalization):

Chuẩn hóa là một quy trình trong đó các giá trị được dịch chuyển và tỷ lệ giữa 0 và 1. Còn được gọi là Min-Max Scaling. Nó được sử dụng để chuẩn hóa dữ liệu sao cho các đặc điểm có một tỷ lệ tương tự. 

Với công thức: 		                                                   (3.2.1.1)

+ Trong đó Xmin là giá trị tối thiểu và Xmax là giá trị tối đa của đặc điểm.
Chuẩn hóa là phương pháp tối ưu khi tập dữ liệu không có Phân phối Gaussian. Hữu ích cho các thuật toán như K─NN mà không giả định bất kỳ phân phối nào.

.........................................................................................

from sklearn.preprocessing import MinMaxScaler

- Tạo một đối tượng của StandardScaler	

 scaler = MinMaxScaler()

- Khớp (fit) dữ liệu của dataframe với scaler print(scaler.fit(features_data))

- Chuyển đổi dataframe

features_scaled = scaler.transform(features_data)

- Chuyển đổi mảng đã được chuẩn hóa thành dataframe

features_scaled = pd.DataFrame(features_scaled, columns=features_data.columns)

......................................................................................

3.2.2.	Thể hiện biểu đồ:

3.2.2.1. Biểu đồ Violin:

Nhóm 10 đặc điểm để quan sát dữ liệu số ➩ so sánh phân phối giữa nhiều nhóm:

a.	10 đặc trưng đầu tiên:

Hình 3.2.2.1. 1: Biểu đồ Violin thể hiện 10 đặc trưng đầu tiên

Từ biểu đồ trên, thấy rằng một số đặc điểm như radius_mean và texture_mean có phân phối tương tự, trong đó giá trị trung bình của Lành tính được tách biệt khỏi Ác tính. Những đặc điểm như vậy có thể phù hợp cho việc phân loại.

Trong khi đó, đặc điểm fractal_dimension_mean có giá trị trung bình của Lành tính và Ác tính gần như giống nhau, do đó không phù hợp để sử dụng đặc điểm này cho việc phân loại.
 
b.	10 đặc trưng tiếp theo:
 
Hình 3.2.2.1. 2: Biểu đồ Violin thể hiện 10 đặc trưng tiếp theo

c.	Các đặc trưng còn lại 
 
Hình 3.2.2.1. 3: Biểu đồ Violin thể hiện các đặc trưng còn lại
Các đặc điểm concavity_worst và concave point_worst có vẻ giống nhau, để tốt hơn về phân phối (nếu các đặc điểm có mối tương quan với nhau) một trong các đặc điểm có thể bị loại bỏ.
3.2.2.2. Biểu đồ Jointplot , Biểu đồ Tương quan bằng Heatmap, Biểu đồ Swarm:
a. Thể hiện biểu đồ Jointplot:
# Biểu đồ Jointplot cho hai đặc điểm
graph = sns.jointplot(x="concavity_worst", y="concave points_worst", data=features_scaled, kind="reg")
r, p = stats.pearsonr(x=features_scaled['concavity_worst'], y=features_scaled['concave points_worst'])
phantom, = graph.ax_joint.plot([], [], linestyle="", alpha=0)
graph.ax_joint.legend([phantom],['r={:f}, p={:f}'.format(r,p)])
Kết quả:
 
Hình 3.2.2.2.a. 1: Biểu đồ Jointplot cho hai đặc điểm “concavity_worst” & “concave points_worst”
Từ Biểu đồ Joint Plot trên, phân phối dữ liệu của các đặc điểm trên khá tương tự. Giá trị Pearsonr là 0.85, gần bằng 1.0, cho thấy sự tương quan mạnh mẽ giữa các đặc điểm concavity_worst và concave point_worst. Do đó, có thể loại bỏ đặc điểm concavity_worst và concave point_worst khỏi tập dữ liệu vì chúng tương quan với nhau.
─ Giới thiệu thuật ngữ:
•	r (hệ số tương quan Pearson):
+ r đo lường mức độ tương quan tuyến tính giữa hai biến. Nó có giá trị từ -1 đến 1.
+ Giá trị r gần 1 cho thấy có mối quan hệ tuyến tính dương mạnh giữa hai biến, trong khi r gần -1 cho thấy có mối quan hệ tuyến tính âm mạnh.
+ Nếu r gần 0, đó có thể là dấu hiệu của sự không tương quan hoặc mối quan hệ không tuyến tính.
•	p (giá trị p):
+ p đo lường mức độ tin cậy của giá trị r. Nếu p nhỏ (thường dưới 0.05), bạn có thể kết luận rằng có một mối quan hệ tuyến tính có ý nghĩa giữa hai biến.
+ Giá trị p lớn có thể chỉ ra rằng không có đủ bằng chứng để bác bỏ giả thuyết không có mối quan hệ.
•	Legend:
+ Đoạn mã cuối cùng tạo một đối tượng legend trên biểu đồ để hiển thị giá trị r và p.
+ r được hiển thị trên biểu đồ để thể hiện mức độ mối quan hệ tuyến tính.
+ p không hiển thị trực tiếp trên biểu đồ, nhưng thông thường được sử dụng để đưa ra quyết định về sự ý nghĩa thống kê của mối quan hệ.
b. Thể hiện biểu đồ Tương quan  Heatmap:
# Biểu đồ Tương quan bằng Heatmap
f,ax = plt.subplots(figsize=(20,25))
sns.heatmap(features_scaled.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
Kết quả:
Từ biểu đồ Heat Map Tương quan bên dưới, có thể thấy rằng các đặc điểm radius_mean, perimeter_mean, area_mean có mối tương quan với nhau.
Tương tự, các đặc điểm compactness_mean, concavity_mean, concave points_mean cũng có mối tương quan với nhau.
 
Hình 3.2.2.2.b. 1: biểu đồ Tương quan  Heatmap
 c. Thể hiện biểu đồ Swarm:
─ Giới thiệu thuật ngữ:
+ plt.figure(figsize=(15, 8)): tạo một hình vẽ với kích thước cụ thể (15x8).
+ sns.swarmplot(): sử dụng thư viện seaborn để tạo biểu đồ Swarm.
+ data=swarm_data: Dữ liệu được truyền vào biểu đồ.
+ x="features": Đặc điểm trên trục x, là tên các đặc trưng.
+ y="value": Giá trị của đặc trưng trên trục y.
+ palette="Greens": Màu sắc của các điểm trên biểu đồ, sử dụng bảng màu "Greens".
+ hue="diagnosis": Sử dụng màu sắc để phân biệt giữa các nhóm chẩn đoán (ác tính và lành tính).

# Biểu đồ Swarm
swarm_data = pd.concat([features_scaled[["radius_mean","perimeter_mean","area_mean",
    "compactness_mean", "concavity_mean", "concave points_mean"]], labels_data], axis=1)
swarm_data = pd.melt(swarm_data, id_vars="diagnosis" , var_name='features', value_name='value')
plt.figure(figsize=(15,8))
sns.swarmplot(data=swarm_data, x="features", y="value" ,palette="Set2", hue="diagnosis")

Kết quả:
 
Hình 3.2.2.2.c. 1: Biểu đồ Swarm của các thuộc tính “radius_mean”,”perimeter_mean”,”area_mean”,
“compactness_mean”, “concavity_mean”, “concave points_mean”

# Biểu đồ Swarm
swarm_data = pd.concat([features_scaled[["radius_se","perimeter_se","area_se", "radius_worst",
    "perimeter_worst", "area_worst", 'texture_mean']], labels_data], axis=1)
swarm_data = pd.melt(swarm_data, id_vars="diagnosis" , var_name='features', value_name='value')
plt.figure(figsize=(15,8))
sns.swarmplot(data=swarm_data, x="features", y="value" ,palette="Set2", hue="diagnosis")

Kết quả:
 
Hình 3.2.2.2.c. 2: Biểu đồ Swarm của các thuộc tính “radius_se”,“perimeter_se”, “area_se”,“radius_worst”,“perimeter_worst”,“area_worst”,“texture_mean”
d.	Nhận xét từ biểu đồ:
•	Từ biểu đồ Heatmap, có thể thấy các đặc điểm radius_mean, perimeter_mean, area_mean có mối tương quan với nhau, vì vậy có thể sử dụng area_mean.
Tương tự,
•	compactness_mean, concavity_mean, concave points_mean cũng tương quan với nhau, vì vậy có thể sử dụng concavity_mean.
•	radius_se, perimeter_se và area_se có mối tương quan, do đó, loại bỏ tất cả các đặc điểm ngoại trừ area_se.
•	radius_worst, perimeter_worst, area_worst và area_worst được chọn cho mục đích phân loại.
➪ Sử dụng cột thuộc tính: area_mean với texture_mean, area_worst
Câu hỏi đặt ra: 
─ Tại sao chỉ chọn một số đặc điểm cụ thể từ các đặc điểm tương quan khác nhau?
+ Từ biểu đồ Swarm trên, có thể thấy rằng area_mean là đặc điểm riêng biệt và phân tán rộng, các đặc điểm có cùng tính chất phân phối sẽ không hữu ích cho việc phân loại.
+ Với lý do này, chỉ cần sử dụng các đặc điểm cần thiết.
─ Tại sao lựa chọn đặc điểm?
+ Lựa chọn đặc điểm là một kỹ thuật để chọn ra các đặc điểm quan trọng nhất từ một tập dữ liệu.
+ Quá trình này giảm số lượng biến đầu vào và do đó giảm sự phức tạp của mô hình.
─ Lựa chọn đặc điểm cho K─NN
+ Các thuật toán dựa trên khoảng cách như K-NN, K-means và SVM bị ảnh hưởng nhiều bởi phạm vi của các đặc điểm trong tập dữ liệu.
+ Những thuật toán này dựa trên độ đo khoảng cách, sử dụng khoảng cách giữa các điểm dữ liệu để xác định độ tương tự. Do đó, các đặc điểm liên quan cung cấp độ chính xác tốt hơn.








# Danh sách các đặc điểm tương quan, dự kiến loại bỏ
drop_list = ['perimeter_mean','radius_mean','compactness_mean','concave points_mean','radius_se',
    'perimeter_se','radius_worst','perimeter_worst','compactness_worst','concave points_worst',
    'compactness_se','concave points_se','texture_worst','area_worst']
# Bộ dữ liệu về đặc điểm đã được tỷ lệ lại
features_updated = features_scaled.drop(drop_list,axis = 1)
features_updated.head()
# In kích thước sau khi loại bỏ các đặc điểm tương quan
print("Kích thước sau khi loại bỏ các đặc điểm tương quan:", features_updated.shape)

Kết quả: 
Kích thước sau khi loại bỏ các đặc điểm tương quan: (569, 16)






CHƯƠNG 4: QUY TRÌNH ĐÀO TẠO VÀ ĐÁNH GIÁ MÔ HÌNH
4.1. Chia tập huấn luyện và thử nghiệm (Train-Test Split):
•	Chia tập dữ liệu thành tập huấn luyện và tập thử nghiệm bằng cách sử dụng train_test_split từ mô-đun sklearn.model_selection.
•	Trong trường hợp này, kích thước thử nghiệm được đặt là 0.35 ➪ 35% tập dữ liệu sẽ được sử dụng để kiểm tra hiệu suất mô hình.

from sklearn.model_selection import train_test_split
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features_updated, labels_data, 
test_size=0.35, random_state=42)
4.2 Bộ phân loại K─NN (K─NN Classifier):
•	Bộ phân loại K─NN là một thuật toán phân loại sử dụng khoảng cách giữa các điểm dữ liệu huấn luyện và các điểm dữ liệu thử nghiệm để xác định những điểm dữ liệu giống nhau nhất.
•	Theo mặc định, K-NN sử dụng khoảng cách Euclidean làm độ đo khoảng cách.
Khoảng cách Euclidean:       giữa hai điểm được đo với công thức:
                                                       (4.2.1)
Trong đó:     d(x, y): là khoảng cách Euclidean giữa hai điểm (x) và (y).
(n):  là số chiều trong không gian (số lượng thành phần).
(x_i) và (y_i):  là các thành phần tương ứng của hai điểm (x) và (y).
Sử dụng cho bất kỳ số chiều nào trong khoảng cách Euclidean giữa hai điểm không gian nhiều chiều.
 
Hình 4.2. 1: Khoảng cách Euclidean giữa hai điểm không gian nhiều chiều

Hyperparameter trong K─NN:
	Hyperparameters là các tham số có thể điều chỉnh có thể điều chỉnh để cải thiện hiệu suất của mô hình.
	Các hyperparameter cần xem xét trong K─NN là number of neighbors(số lân cận), weights (trọng số), và distance metric (thước đo khoảng cách). Tham số có thể điều chỉnh phổ biến và đơn giản nhất là number of neighbors(số lân cận).
	Tham số n_neighbors được đặt là 5 (giá trị mặc định).
	Mô hình K─NN sử dụng number of neighbors (số lân cận), weights (trọng số), và distance metric (thước đo khoảng cách) bằng cách sử dụng GridSearchCV từ mô-đun sklearn.model_selection
Sau khi đã chia tập dữ liệu, một bộ phân loại K─Nearest Neighbors (K─NN) đã được triển khai để phân loại dữ liệu. Trong trường hợp này, một mô hình với số lân cận là 5 đã được chọn để huấn luyện trên tập huấn luyện (X_train, y_train). Mô hình sau đó được sử dụng để dự đoán nhãn trên tập thử nghiệm (X_test), và kết quả dự đoán được lưu trong biến pred.

from sklearn.neighbors import KNeighborsClassifier
## KNN Classifier with K=5 (Ban đầu)
knn = KNeighborsClassifier(n_neighbors=5)
## Fit mô hình
knn.fit(X_train, y_train)
## Dự đoán các giá trị
pred = knn.predict(X_test)
4.3 Các thước đo đánh giá (Evaluation Metrics):

Các thước đo đánh giá được nhập từ mô đun sklearn.metrics.
─ Phân Loại: Được sử dụng để đánh giá chất lượng của dự đoán bởi mô hình phân loại. Hiển thị như: độ chính xác(precision), độ nhớ lại(recall), và điểm F1(f1-score) dựa trên từng lớp(class) hoặc nhãn(label basis).
o	Độ chính xác (Precision): Tỷ lệ của các lớp được dự đoán đúng trong tổng số dự đoán cho lớp đó.
o	Độ nhớ lại (Recall): Tỷ lệ của các lớp thực tế được dự đoán đúng trong tổng số lớp thực tế của lớp đó.
o	Điểm F1 (F1-score): Trung bình điều hòa của độ chính xác và độ nhớ lại.

─ Ma trận nhầm lẫn (Confusion Matrix): Là một ma trận N x N dùng để đánh giá mô hình phân loại. Ma trận này so sánh giá trị thực tế với giá trị dự đoán.
o	True Positives (TP): Số lượng dự đoán đúng trong lớp thực tế đó.
o	True Negatives (TN): Số lượng dự đoán đúng ngoài lớp thực tế đó.
o	False Positives (FP): Số lượng dự đoán sai trong lớp thực tế đó.
o	False Negatives (FN): Số lượng dự đoán sai ngoài lớp thực tế đó.


 
Hình 4.3. 1: Mô hình thước đo đánh giá (Evaluation Metrics)

           Điểm Độ Chính Xác (Accuracy Score): Nhận giá trị thực tế và giá trị dự đoán làm đầu vào và trả về độ chính xác của mô hình.
          Điểm Cross Validation (Cross Val Score): Nhận giá trị đầu vào là tập dữ liệu và cấu hình kiểm tra chéo, sau đó trả về một danh sách các điểm độ chính xác cho mỗi lượt kiểm tra chéo.
-	Mỗi lượt kiểm tra chéo là một tập huấn luyện và thử nghiệm.
 
Hình 4.3. 2: Mô hình kiểm định chéo 4 lớp (4-fold validation, với k=4)
Mô tả: 
• Fold 1:
+ Testing set: Phần dữ liệu được giữ ra để kiểm thử mô hình (25%).
+ Training set: Các phần còn lại của dữ liệu được sử dụng để đào tạo mô hình (75%).
• Fold 2:
+ Training set: Phần dữ liệu được giữ ra trong Fold 1 được thêm vào training set (50%).
+ Testing set: Phần dữ liệu mới được giữ ra để kiểm thử mô hình (25%).
• Fold 3:
+ Training set: Phần dữ liệu của Fold 1 và Fold 2 được thêm vào training set (75%).
+ Testing set: Phần dữ liệu mới được giữ ra để kiểm thử mô hình (25%).
• Fold 4:
+ Testing set: Phần dữ liệu của Fold 3 được giữ ra để kiểm thử mô hình (25%).
+ Training set: Tất cả dữ liệu đã sử dụng trước đó được sử dụng để đào tạo mô hình (100%).


# Phân Loại và Ma trận nhầm lẫn(Confusion)from sklearn.metrics import classification_report, confusion_matrix
# Độ chính xác
from sklearn.metrics import accuracy_score
# Điểm đánh giá Cross Validation
from sklearn.model_selection import cross_val_score
# Phân Loại và Ma trận nhầm lẫn
print("Phân Loại\n",classification_report(y_test, pred),
"\n\nMa trận nhầm lẫn(Confusion)\n",confusion_matrix(y_test, pred))




Kết quả: 
Phân Loại
                     precision    recall     f1-score     support
        B            0.95           0.95         0.95          129
        M           0.92           0.92         0.92            71
    accuracy                                       0.94          200
   macro avg    0.93          0.93          0.93          200
weighted avg   0.94         0.94          0.94           200
Ma trận nhầm lẫn(Confusion)
 [[123   6]
 [  6  65]]
# Điểm Chính Xác cho Toàn Bộ Dữ Liệu Kiểm Tra
print("Điểm Chính Xác", accuracy_score(y_test, pred))
Kết quả: 
Kết quả với điểm chính xác = 94% 
Tiếp tục đào tạo mô hình trên 4 phần và kiểm tra nó trên phần còn lại, sau đó lặp lại quy trình 5 lần với các phần khác nhau.
# Điểm Chính Xác khi chia dữ liệu thành 5 phần 
print(cross_val_score(knn, X_train, y_train, cv=5)) 
Kết quả: 
[0.98648649 0.93243243 0.94594595 0.94594595 0.95890411]
4.3.1 Huấn luyện mô hình biến đổi theo các tỉ lệ:

Một thử nghiệm đơn giản bằng cách thay đổi lượng dữ liệu huấn luyện và kiểm tra và xem xét cách mô hình hoạt động với neighbors mặc định là 5.
Thử nghiệm 1 - Dữ liệu huấn luyện (Train) 67% và dữ liệu kiểm tra (Test) 33%
Thử nghiệm 2 - Dữ liệu huấn luyện (Train) 80% và dữ liệu kiểm tra (Test) 20%
Thử nghiệm 3 - Dữ liệu huấn luyện (Train) 50% và dữ liệu kiểm tra (Test) 50%
a.	Thử nghiệm 1:
# Dữ liệu huấn luyện(Train) 67% và dữ liệu kiểm tra(Test) 33%
X_train, X_test, y_train, y_test = train_test_split(features_updated, labels_data, train_size=0.67, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
## Fit the model
knn.fit(X_train, y_train)
## Dự đoán các giá trị
pred = knn.predict(X_test)
print("Thử nghiệm 1 - Dữ liệu huấn luyện 67% và dữ liệu kiểm tra 33%\n\n Phân loại\n",
classification_report(y_test, pred), "\n\nMa trận nhầm lẫn(Confusion)\n",confusion_matrix(y_test, pred))
print("\nĐiểm chính xác",accuracy_score(y_test, pred))
Kết quả: 
Thử nghiệm 1 - Dữ liệu huấn luyện 67% và dữ liệu kiểm tra 33%
 Phân loại:
                     precision    recall  f1-score   support
           B            0.95      0.95         0.95       121
           M           0.91      0.91         0.91        67
    accuracy                                    0.94       188
   macro avg      0.93     0.93         0.93        188
weighted avg    0.94     0.94          0.94       188
Ma trận nhầm lẫn(Confusion):
 [[115   6]
 [  6  61]]
Điểm chính xác: 0.9361702127659575
b.	Thử nghiệm 2:
# Dữ liệu huấn luyện(Train) 80% và dữ liệu kiểm tra(Test) 20%
X_train, X_test, y_train, y_test = train_test_split(features_updated, labels_data, train_size=0.80, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
## Fit the model
knn.fit(X_train, y_train)
## Dự đoán các giá trị
pred = knn.predict(X_test)
print("Thử nghiệm 2 - Dữ liệu huấn luyện 80% và dữ liệu kiểm tra 20%\n\nPhân loại\n",
classification_report(y_test, pred), "\n\nMa trận nhầm lẫn(Confusion)\n",confusion_matrix(y_test, pred))
print("\nĐiểm chính xác",accuracy_score(y_test, pred))
Kết quả: 
Thử nghiệm 2 - Dữ liệu huấn luyện 80% và dữ liệu kiểm tra 20%
Phân loại:
                     precision    recall  f1-score   support
        B                0.94       0.94      0.94          71
        M               0.91       0.91      0.91          43
    accuracy                                   0.93          114
   macro avg       0.93      0.93      0.93          114
weighted avg     0.93      0.93      0.93          114
Ma trận nhầm lẫn(Confusion):
 [[67  4]
 [ 4 39]]
Điểm chính xác: 0.9298245614035088
c.	Thử nghiệm 3:
# Dữ liệu huấn luyện(Train) 50% và dữ liệu kiểm tra(Test) 50%
X_train, X_test, y_train, y_test = train_test_split(features_updated, labels_data, train_size=0.50, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
## Fit the model
knn.fit(X_train, y_train)
## Dự đoán các giá trị
pred = knn.predict(X_test)
print("Thử nghiệm 3 - Dữ liệu huấn luyện 50% và dữ liệu kiểm tra 50%\n\nPhân loại\n",
classification_report(y_test, pred), "\n\nMa trận nhầm lẫn(Confusion)\n",confusion_matrix(y_test, pred))
print("\nĐiểm chính xác",accuracy_score(y_test, pred))
Kết quả: 
Thử nghiệm 3 - Dữ liệu huấn luyện 50% và dữ liệu kiểm tra 50%
Phân loại:
                      precision    recall  f1-score   support
       B                 0.96        0.96      0.96         187
       M                0.93        0.92      0.92          98
    accuracy                                    0.95          285
   macro avg       0.94        0.94      0.94          285
weighted avg      0.95       0.95       0.95         285
Ma trận nhầm lẫn(Confusion):
 [[180   7]
 [  8  90]]
Điểm chính xác: 0.9473684210526315
d.	Kết luận:
Với một neighbors mặc định là 5, kết quả của các thử nghiệm trên là như sau:
Bảng 4.3.1.d. 1: Kết quả huấn luyện từ 3 thử nghiệm
Thử nghiệm	Dữ liệu Huấn luyện	Dữ liệu Kiểm tra	Điểm chính xác
1	67	33	0.936
2	80	20	0.929
3	50	50	0.947

•	Từ quan sát trên, Có thể thấy rằng điểm chính xác thay đổi khi dữ liệu huấn luyện và dữ liệu kiểm tra được biến đổi.
•	Theo lý tưởng, một tỷ lệ nhỏ hơn dữ liệu huấn luyện sẽ giúp mô hình học tốt hơn. Và một tỷ lệ nhỏ hơn của dữ liệu kiểm tra sẽ giúp mô hình đánh giá mô hình.
Câu hỏi đặt ra:
─ Làm thế nào để cấu hình chia dữ liệu huấn luyện và kiểm tra?
•	Không có phần trăm chia dữ liệu huấn luyện và kiểm tra tốt nhất cho mọi tình huống.
•	Việc chia dữ liệu phụ thuộc vào nhiều yếu tố như:
o	Chi phí tính toán trong việc huấn luyện và đánh giá mô hình
o	Số lượng lớp trong tập dữ liệu huấn luyện và kiểm tra
─ Phương pháp xác nhận tốt là gì?
•	Trong phân loại ung thư vú ─ KNN, tập dữ liệu mất cân bằng. Cụ thể, số lượng mẫu lành tính nhiều hơn nhiều so với số lượng mẫu ác tính.
o	Số lượng lành tính: 357 và Ác tính: 212 
•	Khi dữ liệu được phân chia, có khả năng loại điểm dữ liệu cụ thể có thể được đưa vào tập dữ liệu huấn luyện hoặc thử nghiệm. Điều này dẫn đến các vấn đề như Overfitting và Underfitting.
•	Vì vậy cần phải cân bằng tập dữ liệu. Chia tập dữ liệu sao cho duy trì tỷ lệ mẫu lành tính và ác tính có tỉ lệ bằng nhau (equal proportion).
•	Được thực hiện bằng phương pháp Phân chia thử nghiệm đào tạo phân tầng ( Stratified Train─Test Split), trong đó tỷ lệ của cả hai nhãn được duy trì như nhau trong phân chia đào tạo và thử nghiệm.
4.3.2 Phân chia thử nghiệm đào tạo phân tầng (Stratified Train-Test Split):
# Dữ liệu huấn luyện 67% với Stratified Split
X_train, X_test, y_train, y_test = train_test_split(features_updated, labels_data, train_size=0.67,
random_state=42, stratify=labels_data)
knn = KNeighborsClassifier(n_neighbors=5)
## Fit the model
knn.fit(X_train, y_train)
## Dự đoán giá trị
pred = knn.predict(X_test)
print("Chia Dữ liệu huấn luyện và kiểm tra theo phương pháp Stratified\n\nPhân loại\n",classification_report(y_test, pred),
"\n\nMa trận nhầm lẫn(Confusion)\n",confusion_matrix(y_test, pred))
print("\nĐiểm chính xác",accuracy_score(y_test, pred))

Kết quả:
Chia Dữ liệu huấn luyện và kiểm tra theo phương pháp Stratified
Phân loại:
                        precision    recall  f1-score   support
      B                    0.95        0.99      0.97         118
      M                   0.98        0.91      0.95          70
    accuracy                                       0.96         188
   macro avg         0.97        0.95      0.96         188
weighted avg       0.96         0.96      0.96         188
Ma trận nhầm lẫn(Confusion):
 [[117   1]
 [  6  64]]
Điểm chính xác: 0.9627659574468085
Việc sử dụng stratified split của tập dữ liệu đã cải thiện độ chính xác của mô hình phân loại.
•	Khi sử dụng Stratified Split, mô hình có cơ hội tốt hơn để hiểu về tập dữ liệu, vì mỗi phần tử dữ liệu huấn luyện chứa một tỷ lệ cân bằng giữa các nhãn của tập dữ liệu huấn luyện.
4.4 Tinh chỉnh mô hình K─NN, siêu tham số:
•	Tinh chỉnh siêu tham số là quá trình điều chỉnh các siêu tham số của một mô hình để đạt được hiệu suất tốt nhất có thể.
•	Giá trị tối ưu cho các siêu tham số giúp giảm thiểu nhiễu trong phân loại(noise on classification) và hiện tượng (overfitting) của mô hình
•	Tinh chỉnh Siêu tham số được sử dụng hai kỹ thuật:
o	Tìm kiếm Lưới (Grid Search)
o	Phương pháp Elbow (Elbow Method)
4.4.1. Phương pháp Elbow (Elbow Method):
•	Trong Phương pháp Elbow, giá trị tối ưu của n_neighbors được tìm bằng cách biến đổi số lượng cụm từ một phạm vi các giá trị.
•	Tính toán Tổng bình phương bên trong cụm (WCSS) cho mỗi giá trị của n_neighbors hoặc K.
•	WCSS là tổng bình phương khoảng cách giữa trọng tâm của một cụm và từng điểm trong cụm. Với biểu đồ, có thể quan sát đường cong giảm đi một giá trị nhất định, tạo ra một "Elbow Method" trong biểu đồ.
•	Giá trị tương ứng với điểm Elbow Method này là giá trị tối ưu của K.
a.	Tỷ lệ chính xác so với N_Neighbors(Accuracy Rate vs N_Neighbors):
accuracy_rate = []
# phạm vi(Range) của n_neighbors cho KNN
for i in range(1,40):
knn = KNeighborsClassifier(n_neighbors=i)
scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
accuracy_rate.append(scores.mean())
plt.figure(figsize=(12,6))
accuracy_plot = plt.plot(range(1,40), accuracy_rate, color='blue', linestyle='dashed', marker='o',
markerfacecolor='red', markersize=10)
accuracy_plot = plt.title('Tỷ lệ chính xác so với giá trị K')
accuracy_plot = plt.xlabel('N_neighbors')
accuracy_plot = plt.ylabel('Accuracy Rate(Độ chính xác)')
Từ biểu đồ bên dưới , Tỷ lệ chính xác (Accuracy Rate) đang giảm đối với các giá trị cao hơn của N_neighbors. giá trị tối ưu cho n_neighbors từ biểu đồ trên là 11.
 
Hình 4.4.1.a. 1: Tỷ lệ chính xác so với N_Neighbors(Accuracy Rate vs N_Neighbors)
b. Tỷ lệ lỗi so với N_Neighbors (Error Rate vs N_Neighbors):
error_rate = []
# phạm vi(Range) của n_neighbors cho KNN
for i in range(1,50):
knn = KNeighborsClassifier(n_neighbors=i)
scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
error_rate.append(1 - scores.mean())
plt.figure(figsize=(12,6))
error_rate_plot = plt.plot(range(1,50), error_rate, color='blue', linestyle='dashed', marker='o',
markerfacecolor='red', markersize=10)
error_rate_plot = plt.title('Tỷ lệ lỗi so với giá trị K')
error_rate_plot = plt.xlabel('N_neighbors')
error_rate_plot = plt.ylabel('Error Rate(Tỷ lệ lỗi)')
Từ biểu đồ bên dưới, rõ ràng là Tỷ lệ lỗi(Error Rate) đang tăng đối với các giá trị cao hơn của n_neighbors. giá trị tối ưu (optimal value) cho n_neighbors từ biểu đồ trên là 11. phân loại mô hình KNN cho n_neighbors = 11
 
Hình 4.4.1.b. 1: Tỷ lệ lỗi so với N_Neighbors (Error Rate vs N_Neighbors)
c. Thực hiện phân loại mô hình KNN:
 Cho n_neighbors = 11

# Huấn luyện dữ liệu 67% với Phân chia phân tầng
X_train, X_test, y_train, y_test = train_test_split(features_updated, labels_data, train_size=0.67,
random_state=42, stratify=labels_data)
knn = KNeighborsClassifier(n_neighbors=11)
## Phù hợp với mô hình
knn.fit(X_train, y_train)
## Dự đoán các giá trị
pred = knn.predict(X_test)
print("Phân chia phân tầng giữa Tập Huấn luyện(Train) - Tập Kiểm thử(Test)\n\nPhân loại\n", classification_report(y_test, pred),
"\n\nMa trận nhầm lẫn(Confusion)\n", confusion_matrix(y_test, pred))
print("\nĐiểm chính xác(Accuracy Score)", accuracy_score(y_test, pred))

Kết quả:
Phân chia phân tầng giữa Tập Huấn luyện (Train) ─ Tập Kiểm thử (Test)
Phân loại:
                          precision    recall  f1-score   support
      B                    0.95          0.99      0.97         118
      M                   0.98          0.91      0.95            70
    accuracy                                         0.96          188
   macro avg         0.97          0.95       0.96          188
weighted avg       0.96           0.96      0.96          188
Ma trận nhầm lẫn(Confusion):
 [[117   1]
 [  6  64]]
Điểm chính xác(Accuracy Score): 0.9627659574468085
4.4.2. Tìm Kiếm Lưới (Grid Search):
•	Lặp qua một lưới hyperparameter đã được xác định trước và trả về mô hình tốt nhất dựa trên dữ liệu xác thực.
•	Trong kỹ thuật này, các hyperparameter được chia thành các điểm lưới rời rạc và mô hình được huấn luyện trên mỗi điểm lưới đó. Sau đó, mô hình được đánh giá dựa trên các chỉ số hiệu suất.
Khác với Phương pháp Elbow (Elbow Method), Tìm kiếm lưới (Grid Search) có thể được sử dụng để tìm các giá trị tối ưu của nhiều hyperparameter, không chỉ giới hạn ở giá trị n_neighbors.

# Tìm Kiếm Lưới từ Thư viện Lựa chọn Mô hình (Model Selection)
from sklearn.model_selection import GridSearchCV
grid_params = { 'n_neighbors' : [5,7,9,11,13,15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65,
70, 75, 80, 85, 90, 95, 100],
'weights' : ['uniform','distance'],
'metric' : ['minkowski','euclidean','manhattan']}
# Tìm kiếm lưới trên KNN cho 10-fold cross validation
gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose = 1, cv=10, n_jobs = -1)
g_res = gs.fit(X_train, y_train)
#Độ chính xác tốt nhất
g_res.best_score_
Kết quả: 0.958029689608637
#Các tham số tốt nhất 
g_res.best_params_
Kết quả: {'metric': 'manhattan', 'n_neighbors': 11, 'weights': 'uniform'
■ Huấn luyện mô hình với các tham số tốt nhất của mô hình :
# Huấn luyện dữ liệu 67% với Phân chia phân tầng
X_train, X_test, y_train, y_test = train_test_split(features_updated, labels_data, train_size=0.67,
random_state=42, stratify=labels_data)
knn = KNeighborsClassifier(n_neighbors=11, weights='uniform', metric='manhattan')
## Phù hợp với mô hình
knn.fit(X_train, y_train)
## ## Dự đoán các giá trị
pred = knn.predict(X_test)
print("Phân chia phân tầng giữa Tập Huấn luyện(Train) - Tập Kiểm thử(Test)\n\nPhân loại\n", classification_report(y_test, pred),
"\n\nMa trận nhầm lẫn(Confusion)\n", confusion_matrix(y_test, pred))
print("\nĐiểm chính xác(Accuracy Score)", accuracy_score(y_test, pred))
Kết quả:
Phân chia phân tầng giữa Tập Huấn luyện(Train) - Tập Kiểm thử(Test)
Phân loại
                          precision    recall  f1-score   support
       B                    0.94         1.00      0.97         118
       M                   1.00         0.90      0.95          70
    accuracy                                         0.96         188
   macro avg           0.97        0.95      0.96          188
weighted avg         0.96         0.96      0.96         188
Ma trận nhầm lẫn(Confusion):
 [[118   0]
 [  7  63]]
Điểm chính xác(Accuracy Score): 0.9627659574468085
4.5 Kết quả và thảo luận:
•	Đối với Thử nghiệm 2 đã triển khai thành công phân loại dựa trên K─NN cho ung thư vú.
•	Breast Cancer dataset (Bộ dữ liệu ung thư vú) là một bộ dữ liệu mất cân bằng với 357 mẫu lành tính và 212 mẫu ác tính.
•	Việc khám phá và trực quan hóa dữ liệu chi tiết được thực hiện lựa chọn đặc trưng, vì bộ dữ liệu có nhiều đặc trưng có sự tương quan.
•	Các thử nghiệm dựa trên kích thước mẫu biến đổi đã được tiến hành để quan sát sự biến đổi trong hiệu suất của bộ phân loại.
•	Việc phân chia dữ liệu phân tầng là lựa chọn tối ưu cho bộ dữ liệu mất cân bằng vì nó tránh các vấn đề của Overfitting và Underfitting, bằng cách đảm bảo tỉ lệ của các nhãn trong mỗi lượt là bằng nhau. Điều này đã làm tăng độ chính xác của mô hình từ 0.95 lên 0.96.
•	Tối ưu hoá Hyperparameter đã được thực hiện bằng cách sử dụng Tìm kiếm lưới và kỹ thuật Phương pháp Elbow để so sánh. Khác với phương pháp Elbow, tìm kiếm lưới có thể được sử dụng để tìm các giá trị tối ưu của nhiều hyperparameter, không chỉ giới hạn ở giá trị n_neighbors.
•	Các Thước đo đánh giá có liên quan như ma trận nhầm lẫn, phân loại, độ chính xác, F1-Score được tính toán cho mô hình K─NN.

 
CHƯƠNG 5: TRIỂN KHAI ỨNG DỤNG & PHƯƠNG HƯỚNG PHÁT TRIỂN
5.1 Cài đặt và sử dụng:
─ Cài đặt Visual Studio Code, Visual Studio, PyCharm…
─ Cài đặt thư viện streamlit
─  Load file chương trình, vào terminal gõ lệnh: streamlit run app.py
                          (chương trình có tên “app.py”)
 
Hình 5.1. 1: kết quả sau khi vào terminal gõ lênh “streamlit run app.py”
─ Sau đó chương trình xuất hiện giao diện như sau: 
 
Hình 5.1. 2: giao diện chẩn đoán mắc ung thư Ác tính


 
Hình 5.1. 3: giao diện chẩn đoán mắc ung thư Ác tính



5.2 Những thách thức và hướng phát triển:
a. Thách Thức Hiện Tại: 
Chương trình hiện tại đưa ra kết quả về việc mắc ung thư vú chỉ dưới dạng lành tính hoặc ác tính, không nêu rõ các thông tin chi tiết như giai đoạn của bệnh và còn thiếu khả năng nhận biết trường hợp người hoàn toàn không mắc bệnh. Điều này có thể tạo ra một hạn chế đối với các chuyên gia y tế và người dùng khi muốn có cái nhìn chi tiết hơn về tình trạng sức khỏe của bệnh nhân.
b. Hướng Phát Triển và Cải Thiện:
Chi Tiết Hóa Giai Đoạn của Bệnh:
•	Mục tiêu cụ thể là mở rộng chương trình để cung cấp thông tin chi tiết hơn về các giai đoạn của ung thư vú. Điều này có thể giúp bác sĩ và chuyên gia y tế đưa ra quyết định chẩn đoán và điều trị hiệu quả hơn.
Nhận Biết Bệnh Nhân Hoàn Toàn Khỏe Mạnh:
•	Bổ sung khả năng nhận biết trường hợp người hoàn toàn không mắc bệnh. Điều này quan trọng để tránh việc gây lo lắng không cần thiết cho những người không mắc bệnh và giúp tăng độ tin cậy của chương trình.
Giao Diện Thân Thiện và Thao Tác Đơn Giản:
•	Cải thiện giao diện người dùng để làm cho quá trình thao tác đơn giản và thân thiện hơn. Điều này có thể bao gồm việc tối ưu hóa hiển thị kết quả, thêm hướng dẫn sử dụng, và cải thiện trải nghiệm người dùng.
Mở Rộng Đối Với Các Trường Hợp Ngoại Lệ:
•	Mở rộng chương trình để xử lý các trường hợp ngoại lệ khác như bệnh nhân mắc ở các giai đoạn cụ thể (1, 2, 3,...) theo tình hình thực tế. Điều này giúp chương trình trở nên linh hoạt và đáp ứng được nhiều tình huống khác nhau.
Kết Luận: 
Chúng em hy vọng rằng những cải tiến và mở rộng với đề xuất trên sẽ giúp chương trình trở thành một công cụ hữu ích và chính xác hơn trong việc chẩn đoán ung thư vú. Việc chi tiết hóa thông tin, nhận biết trường hợp người hoàn toàn không mắc bệnh, và tối ưu hóa giao diện sẽ làm cho chương trình trở nên mạnh mẽ và thân thiện hơn đối với người dùng cuối và các chuyên gia y tế.
 

