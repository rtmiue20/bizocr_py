import numpy as np
from sklearn.linear_model import LinearRegression


class AnomalyService:
    def detect_amount_anomaly(self, current_amount: float, historical_amounts: list) -> dict:
        """Kiểm tra bất thường của số tiền dựa trên Lịch sử"""
        if not historical_amounts or len(historical_amounts) < 3:
            return {
                "is_anomaly": False,
                "reason": "Chưa đủ dữ liệu lịch sử để phân tích (cần ít nhất 3 mẫu)"
            }

        try:
            # 1. Thuật toán Mean & Standard Deviation (Z-Score)
            mean_val = np.mean(historical_amounts)
            std_val = np.std(historical_amounts)
            z_score = abs(current_amount - mean_val) / std_val if std_val > 0 else 0

            # 2. Thuật toán Median & Median Absolute Deviation (MAD) - Chống nhiễu tốt hơn Mean
            median_val = np.median(historical_amounts)
            mad = np.median([abs(x - median_val) for x in historical_amounts])
            modified_z_score = 0.6745 * abs(current_amount - median_val) / mad if mad > 0 else 0

            # 3. Thuật toán Regression (Hồi quy tuyến tính) - Dự đoán xu hướng
            X = np.array(range(len(historical_amounts))).reshape(-1, 1)
            y = np.array(historical_amounts)
            reg = LinearRegression().fit(X, y)
            expected_trend_val = reg.predict(np.array([[len(historical_amounts)]]))[0]

            # Tính độ lệch so với xu hướng hồi quy
            trend_deviation = abs(current_amount - expected_trend_val) / (
                expected_trend_val if expected_trend_val > 0 else 1)

            # --- Quyết định Outlier (Bất thường) ---
            # Bất thường nếu Z-score quá cao, Modified Z-score quá cao, hoặc chệch quá 50% so với Regression Trend
            is_anomaly = bool(z_score > 3.0 or modified_z_score > 3.5 or trend_deviation > 0.5)

            return {
                "is_anomaly": is_anomaly,
                "metrics": {
                    "mean": float(mean_val),
                    "median": float(median_val),
                    "expected_trend": float(expected_trend_val)
                },
                "reason": "Số tiền vượt xa phân phối thống kê hoặc xu hướng lịch sử" if is_anomaly else "Hợp lệ"
            }
        except Exception as e:
            return {"is_anomaly": False, "reason": f"Lỗi tính toán ML: {str(e)}"}


anomaly_service = AnomalyService()