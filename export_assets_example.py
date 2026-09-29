"""ตัวอย่างโค้ดสำหรับวางท้าย notebook ที่เทรนโมเดล เพื่อสร้างไฟล์ที่แอปใช้
(ไม่ต้องอัปขึ้น GitHub ก็ได้ — รันในเครื่องแล้วนำไฟล์ผลลัพธ์ไปวางข้าง app.py)

สมมติตัวแปรจาก notebook ของคุณ:
  scaler      = StandardScaler ที่ fit กับชุดเทรน (ถ้ามี)
  X_test_raw  = DataFrame ชุดทดสอบ 'ก่อน' scale (คอลัมน์ Time, V1..V28, Amount)
  y_test      = ค่า Class ของชุดทดสอบ
"""
import joblib

# 1) บันทึก scaler (ข้ามได้ถ้าไม่ได้ scale ตอนเทรน)
joblib.dump(scaler, "scaler.pkl")

# 2) บันทึกชุดทดสอบ โดยต้องเป็นค่าดิบที่ยังไม่ scale
test_df = X_test_raw.copy()
test_df["Class"] = list(y_test)
test_df.to_csv("test_data.csv", index=False)
