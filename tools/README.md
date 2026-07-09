# Accuracy testing

วัด **ความแม่นยำ** ของตัวสแกน (Thai ID / passport) โดยยิงรูปจริงเข้า API แล้วเทียบค่า 6 fields กับเฉลย (ground truth) — ไม่ใช่แค่เช็ค schema เหมือน `tests/test_integration.py`

`accuracy.py` เป็น **pure stdlib** (ไม่ต้อง pip install อะไรเพิ่ม) ยิงผ่าน HTTP แบบเดียวกับ curl / NestJS backend

## ขั้นตอน

1. **เตรียมภาพ** — วางไฟล์ `.jpg` / `.png` ลงใน `.test-fixtures/` (โฟลเดอร์นี้ git-ignored อยู่แล้ว รูปกับ PII จะไม่หลุดขึ้น git)

2. **เขียนเฉลย** — copy template แล้วกรอกค่าที่ถูกต้องของแต่ละใบ:
   ```bash
   cp tools/labels.example.json .test-fixtures/labels.json
   # แก้ให้ key = ชื่อไฟล์รูป, ใส่ค่าที่ควรอ่านได้ของแต่ละ field
   ```
   - `type` บังคับ: `thai_id` หรือ `passport`
   - field ไหนไม่อยากให้นับ ให้ลบทิ้งหรือใส่ `null`
   - `date_of_birth` เป็น ISO `YYYY-MM-DD`, `sex` เป็น `M`/`F`, `country` เป็น ISO-3 (`THA`)

3. **รันเซอร์วิส** (Thai ID ต้องใช้ PaddleOCR → รันใน docker):
   ```bash
   docker compose build && docker compose up -d
   ```

4. **วัดผล:**
   ```bash
   export API_KEY=$(grep API_KEY .env | cut -d= -f2)
   python tools/accuracy.py --labels .test-fixtures/labels.json --url http://localhost:8000
   ```

## ผลลัพธ์

- **Per-field accuracy** — field ไหนแม่น/พลาดบ่อย (ชื่อไทยมักยากสุด) พร้อมโชว์ค่าที่อ่านได้ vs เฉลยของใบที่ผิด
- **Document accuracy** — สัดส่วนใบที่อ่านถูก **ครบทุก field** (metric ที่ผู้ใช้จริงสัมผัส)

การเทียบ normalize ให้แล้ว (ตัดช่องว่าง/ขีดใน document_number, casefold ชื่อ, ตัดเวลาออกจากวันที่) ความต่างเชิงรูปแบบเลยไม่นับเป็นผิด

## ใช้เป็น regression gate (CI)

ตั้ง threshold แล้ว exit code จะเป็น 1 ถ้าต่ำกว่า — เอาไปใส่ CI กันความแม่นยำตกได้:
```bash
python tools/accuracy.py --labels .test-fixtures/labels.json \
    --min-field 0.90 --min-doc 0.70
```

`--json` สั่งให้ออกผลเป็น JSON ล้วน (เก็บ log / trend ต่อได้)
