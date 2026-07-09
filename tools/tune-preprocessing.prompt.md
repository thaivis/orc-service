# Prompt: วัด + จูน accuracy ของ Thai ID scanner (docker, hill-climb)

> วิธีรัน: `/do-work` แล้วสั่งว่า **"ทำตาม tools/tune-preprocessing.prompt.md"**
> (หรือ copy ทั้ง block ด้านล่างไปวางเป็น task)
>
> สถานะปัจจุบัน (2026-07-08): รูป 8 ใบ + `.test-fixtures/labels.json` เตรียมไว้แล้ว,
> docker image build แล้ว, service รันอยู่ที่ `http://localhost:8000`,
> API_KEY = `test-validation-key-phase1`

---

เป้าหมาย: เพิ่มความแม่นยำการอ่านข้อมูลของ **Thai ID scanner** โดย**จูนเฉพาะค่าตัวเลข**ใน
image preprocessing แล้ววัดผลกับเฉลยจริง — ทำเป็นลูป hill-climb แต่ **baseline ก่อน แล้วหยุดให้คนตัดสิน**

## ขอบเขต (เข้มงวด)
- แก้ได้**เฉพาะค่าคงที่ตัวเลข**: `app/preprocessing.py` (Canny `(50,150)`, CLAHE `clipLimit=2.0`/
  `tileGridSize=(8,8)`, GaussianBlur `(5,5)`, dilate kernel/iterations) + `app/config.py`
  `max_image_dimension`
- **ห้ามแก้ logic** — regex, anchor, การ extract field, MRZ, schema, การเลือกโมเดล OCR ห้ามแตะ
  (ถ้าคิดว่าต้องแก้ logic ถึงจะดีขึ้น → หยุดแล้วรายงาน อย่าแก้เอง)

## รันให้เร็ว + ไม่ค้าง (docker, service รันอยู่แล้ว)
- ใช้ container ที่รันอยู่ + `tools/accuracy.py` ยิงผ่าน HTTP (พิสูจน์แล้วว่าใช้ได้)
- **ทุก request ตั้ง `--timeout 300`** — scan บน Apple Silicon รันผ่าน x86 emulation ช้า (~30-60s/รูป)
  timeout สั้นกว่านี้คือสาเหตุที่ค้าง/ล้มเมื่อก่อน
- **warm โมเดลก่อนเสมอ**: ยิง 1 รูปทิ้ง (`curl --max-time 600 .../scan/thai-id`) ให้โมเดลโหลดจบ
  ก่อนเริ่มวัด — request แรกรวมโหลดโมเดล ~60s, ครั้งถัด ๆ ไปเร็วขึ้น
- **รัน accuracy.py เป็น background job เสมอ** (8 รูป ~5 นาที) อย่ารัน foreground จน block
- **แก้โค้ดแล้วอย่า rebuild** — bind-mount โค้ดครั้งเดียว แล้ว `docker compose restart` (วินาที):
  ก่อนเริ่มลูป เพิ่ม `volumes: [./app:/app/app]` ใน service ของ `docker-compose.yml`
  (เป็น dev override, revert ตอนจบ) เพื่อให้แก้ `preprocessing.py` บน host มีผลใน container
  หลัง restart โดยไม่ต้อง build ใหม่

## Preflight (ยืนยัน ไม่ต้องสร้างใหม่)
1. เช็ค `docker compose ps` ว่า service `Up` — ถ้าไม่ ให้ `docker compose up -d` แล้วรอ `/health` ok
2. `.test-fixtures/labels.json` มีอยู่แล้ว (8 รูป, thai_id ทั้งหมด) — อ่านมาใช้ ไม่ต้องร่างใหม่

## ⚠️ ข้อจำกัดของชุดข้อมูลปัจจุบัน (สำคัญ — อ่านก่อนจูน)
ชุดรูป 8 ใบตอนนี้**ไม่เหมาะกับการจูนจริง** เพราะ:
- `test1.jpg` + `IMG_2729/2730/2731` = **บัตรจริงใบเดียวกัน** (Kittikhun) 4 มุม/การหมุน
- `channarong.jpg` = บัตรจริงอีกใบ (ใช้ได้)
- `nattaya.jpg` / `buarai.jpg` / `demo_sample.jpg` = **รูปเว็บ/เดโมความละเอียดต่ำ** (274-670px)
  จูน preprocessing แก้ไม่ได้ (ข้อมูลตัวอักษรหายตั้งแต่ต้น) และไม่ตรง production (ถ่ายมือถือ)

**แปลว่า: เหมาะกับ "วัด baseline + ดูว่าอะไรพัง" ยังไม่ควรจูน hill-climb เต็มรูปแบบ**
(บัตรจริงต่างใบมีแค่ 2 → แบ่ง tune/holdout ไม่มีความหมาย = overfit ชัวร์)

## Phase 0 — Baseline (ทำอันนี้ก่อน แล้วหยุด)
3. warm โมเดล → รัน `python3 tools/accuracy.py --labels .test-fixtures/labels.json
   --api-key test-validation-key-phase1 --url http://localhost:8000 --timeout 300` (background)
4. รายงาน: field accuracy + document accuracy, ตารางว่ารูปไหน/field ไหนอ่านถูก-ผิด,
   แยกให้เห็นว่า "ผิดเพราะรูปคุณภาพต่ำ (แก้ด้วย preprocessing ไม่ได้)" vs
   "ผิดทั้งที่รูปชัด (อาจแก้ได้)"
5. **หยุด** สรุปให้ผู้ใช้ว่า: baseline เท่าไร, คอขวดอยู่ที่ preprocessing หรือ logic หรือคุณภาพรูป,
   และควรไปต่อทางไหน (เก็บรูปถ่ายมือถือบัตรจริงเพิ่ม / จูน / แก้ logic) — รอผู้ใช้ตัดสินก่อนทำ Phase 1

## Phase 1 — Hill-climb (ทำต่อเมื่อผู้ใช้อนุมัติ + มีรูปถ่ายมือถือบัตรจริง ≥8 ใบต่างคน)
6. แบ่ง tune (70%) / holdout (30%) แบบ deterministic (เรียงชื่อไฟล์ index คู่=tune คี่=holdout)
7. ตัวชี้วัด = **document accuracy บน holdout** (รวมทุกรูป)
8. greedy hill-climb: เปลี่ยนทีละค่า → restart container → warm → วัด →
   ยอมรับเมื่อ tune ดีขึ้น **และ** holdout ไม่ลด → ไม่งั้น revert; เก็บ best เสมอ
9. หยุดเมื่อสแกนครบทุกค่าแล้วไม่มีการยอมรับ (นิ่ง) หรือครบ 10 รอบ

ชุดค่าที่ลองได้ (ห้ามหลุดกรอบ):
- `max_image_dimension`: 1200 / 1600 / 2000 / 2400 / 3000
- CLAHE `clipLimit`: 1.0 / 1.5 / 2.0 / 3.0 / 4.0 ; `tileGridSize`: (4,4) / (8,8) / (16,16)
- GaussianBlur: (3,3) / (5,5) / (7,7) ; Canny: (30,100)/(50,150)/(75,200)/(100,200) ; dilate iters: 1 / 2

## บันทึกผล (git)
10. ทำบน branch ใหม่ (เช่น `tune/preprocessing`) อย่า commit ลง main
11. ทุกค่าที่**ยอมรับ** = 1 commit แยก ระบุ ค่าเก่า→ใหม่ + holdout accuracy ก่อน→หลัง
12. ตอนจบ: revert bind-mount ที่เพิ่มใน docker-compose.yml, รายงาน baseline vs final,
    รูปที่ยังผิด, รัน `pytest -q` เป็น sanity
