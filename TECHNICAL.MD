Chẩn Đoán
# FedAvg: 

chỉ CE + weighted average theo số mẫu ở algorithm/client/fedavg.py (line 77) và algorithm/server/fedavg.py (line 164). Mạnh: nhanh, ổn định. Yếu: không xử lý style/domain shift, nên fairness gap lớn.


# FedProx: 

thêm proximal gradient ở algorithm/client/fedprox.py (line 28). Mạnh: giảm local drift. Yếu: không tạo representation bất biến domain, mu=0.01 khá nhẹ nên gần FedAvg.


# FedSR: 

stochastic feature + L2R/CMI ở algorithm/client/fedsr.py (line 27), model đổi head thành output mu/sigma ở model/models.py (line 153). Mạnh: regularize representation. Yếu: thêm nhiễu/stochastic head, dễ underfit MobileNetV3 với round ít.


# FedIIR: 

align gradient classifier ở algorithm/client/fediir.py (line 58). Mạnh: giảm classifier bias, worst tốt hơn. Yếu: tốn thêm gradient pass, chỉ tác động classifier; còn có điểm đáng sửa là grad_mean khởi tạo zero rồi EMA ngay từ round 0 ở algorithm/server/fediir.py (line 43), nên round đầu kéo gradient về gần 0.


# FedCCRL: 

dùng cross-client style pool, MixStyle + AugMix, JS consistency, supervised contrastive ở algorithm/client/fedccrl.py (line 65). 

Đây là lý do nó tốt: mỗi batch học ảnh gốc + 2 biến thể style, rồi ép prediction/features nhất quán. 

Yếu: chậm hơn FedAvg nhiều vì 3 forward/backward path + AugMix, chưa có fairness/calibration-aware aggregation.

