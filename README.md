# PhanLoaiUngThuVu_BreastCancerClassificationKNN_MachineLearning
Phân Loại Ung Thư Vú Bằng Cách Sử Dụng - KNN 
1.	Importing các thư viện cần thiết
2.	Khám Phá Dữ Liệu
Mô tả về tập dữ liệu
3.	Trực Quan Hóa Dữ Liệu & Lựa Chọn Đặc Điểm
3.1	Chuẩn hóa (Normalization)
# Tạo một đối tượng của StandardScaler	
3.2	Thể hiện biểu đồ:
A.	Biểu đồ Violin theo nhóm 10 đặc điểm để quan sát phân phối dữ liệu số 10 Đặc Điểm Tiếp Theo
Các Đặc Điểm Còn Lại
B.	Biểu đồ Jointplot , Biểu đồ Tương quan bằng Heatmap, Biểu đồ Swarm
4.	Quy trình Đào Tạo
4.1	Chia tập huấn luyện và thử nghiệm (Train-Test Split)
4.2	Bộ phân loại K-NN (K-NN Classifier)
pred = knn.predict(X_test)	
…
…
4.3	Các thước đo đánh giá (Evaluation Metrics)
4.4	Tinh chỉnh Mô hình K-NN - Tối ưu hóa Siêu tham số (Fine-tuning K-NN Model - Hyperparameter Optimization)
A.	Thử nghiệm
	Thử nghiệm 1 - Dữ liệu huấn luyện(Train) 67% và dữ liệu kiểm tra(Test) 33%
	Thử nghiệm 2 - Dữ liệu huấn luyện(Train) 80% và dữ liệu kiểm tra(Test) 20%
	Thử nghiệm 3 - Dữ liệu huấn luyện(Train) 50% và dữ liệu kiểm tra(Test) 50%
B.	Phân chia thử nghiệm đào tạo phân tầng(Stratified Train-Test Split)
C.	Tinh Chỉnh Phương pháp Elbow (Elbow Method)
D.	Tinh Chỉnh Phương pháp Tìm Kiếm Lưới (Grid Search)
5.	Dùng Mô Hình Thể Hiện Biên Quyết Định(Decision Boundary) cho Mô hình K-NN
Code . . .
6.	Tổng kết
