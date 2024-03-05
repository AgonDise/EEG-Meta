# EEG-Meta
# DATA-DESCRIPTION
Data là dạng sóng có 23 channel đã được tách ra và đánh số thứ tự đuôi từ 0~22, tương ứng với các điện cực khác nhau.
folder CHB01 - 24 bao gồm các data được xác định là không xuất hiện động kinh  
folder CHB01 - 24_seizure là các data đã được sàn lọc và chắc chắn có động kinh.

  DATA được lọc bằng Debauchy bậc 6
  Data spectrogram dạng saliency vẫn chưa đạt được như ý muốn

# Model
Model Seizure Detect định hướng dùng Resnet50 để phát hiện đặc trưng, tập trung vào DETECT, không tập trung vào Classitification
Input đầu vào dự định sẽ là (128,256,1) sẽ dùng cả 23 channel.
Model đang được fix lại với data.
