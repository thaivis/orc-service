# orc-service

OCR microservice สำหรับ hotel check-in: รับภาพบัตรประชาชนไทยหรือ passport ต่างประเทศ → คืน JSON 6 fields (first_name / last_name / document_number / date_of_birth / sex / country) ให้ NestJS backend

Self-hosted, ไม่มี cloud OCR / LLM, 0 บาท/scan

ดูข้อกำหนดเต็มที่ [`PRD.md`](./PRD.md) และแผนการ build ที่ [`PLAN.md`](./PLAN.md)

---

## Quick Start

```bash
cp .env.example .env
# เปลี่ยน API_KEY ใน .env เป็น secret string ยาวๆ

docker compose build   # ครั้งแรกใช้เวลานาน — pre-download PaddleOCR Thai+English models
docker compose up -d

curl http://localhost:8000/health
# → {"status":"ok"}
```

ทดสอบ scan:

```bash
curl -X POST http://localhost:8000/scan \
  -H "X-API-Key: <ค่าใน .env>" \
  -F type=passport \
  -F image=@./samples/passport.jpg
```

---

## API

| Method | Path     | Notes                                |
| ------ | -------- | ------------------------------------ |
| `GET`  | `/health` | Health check (ไม่ต้องใช้ API key)     |
| `POST` | `/scan`   | OCR ภาพบัตร — schema ใน `PRD.md §4` |
| `GET`  | `/docs`   | OpenAPI / Swagger UI                  |

ทุก request นอก `/health` ต้องส่ง header `X-API-Key: <secret>` response จะมี header `X-Request-ID` (UUID hex) ใช้ trace log เวลามีปัญหา

### Status codes

| Code | เจอเมื่อ                                                                        |
| ---- | ------------------------------------------------------------------------------ |
| 200  | OCR ทำงานได้, มี fields กลับมา (อาจมี null + low confidence — frontend ตัดสินเอง) |
| 400  | Input malformed (file format / size / type ผิด, decode ไม่ออก)                  |
| 401  | API key ผิด / ไม่ได้ส่ง                                                         |
| 422  | OCR ทำงาน แต่ไม่เจอเอกสาร หรือ detected type ไม่ตรงกับที่ส่งมา                  |
| 500  | Server error                                                                  |

ตัวอย่าง error body:

```json
{ "error": "type_mismatch", "message": "Detected passport but type did not match", "detected_type": "passport" }
```

---

## Calling from NestJS

ใช้ `axios` + `form-data` ส่ง multipart ต่อไปที่ orc-service:

```ts
// src/ocr/ocr.service.ts
import { Injectable, Logger } from '@nestjs/common';
import axios from 'axios';
import FormData from 'form-data';

type ScanType = 'thai_id' | 'passport';

interface ScanResult {
  type: ScanType;
  first_name: string | null;
  last_name: string | null;
  document_number: string | null;
  date_of_birth: string | null;
  sex: 'M' | 'F' | null;
  country: string | null;
  document_valid: boolean;
  confidence: Record<string, number>;
  warnings: string[];
}

@Injectable()
export class OcrService {
  private readonly logger = new Logger(OcrService.name);
  private readonly base = process.env.ORC_SERVICE_URL ?? 'http://orc-service:8000';
  private readonly apiKey = process.env.ORC_SERVICE_API_KEY!;

  async scan(type: ScanType, image: Buffer, filename: string): Promise<ScanResult> {
    const form = new FormData();
    form.append('type', type);
    form.append('image', image, { filename });

    const res = await axios.post<ScanResult>(`${this.base}/scan`, form, {
      headers: { ...form.getHeaders(), 'X-API-Key': this.apiKey },
      timeout: 5000,
      validateStatus: () => true,
    });

    if (res.status !== 200) {
      this.logger.warn(`orc-service ${res.status} requestId=${res.headers['x-request-id']}`);
      throw new Error(`orc-service responded ${res.status}: ${JSON.stringify(res.data)}`);
    }
    return res.data;
  }
}
```

หมายเหตุ NestJS รัน container เดียวกับ orc-service ใน docker-compose จะใช้ hostname `orc-service:8000`

---

## Configuration

| Env var               | Default | Notes                                              |
| --------------------- | ------- | -------------------------------------------------- |
| `API_KEY`             | _none_  | required — shared secret ระหว่าง NestJS ↔ orc      |
| `PORT`                | `8000`  | container internal port                             |
| `MAX_IMAGE_SIZE_MB`   | `10`    | reject ภาพที่ใหญ่กว่านี้ด้วย 400                    |
| `MAX_IMAGE_DIMENSION` | `2000`  | downscale ภาพที่ longest side > นี้ก่อน OCR        |
| `LOG_LEVEL`           | `info`  | `debug` / `info` / `warning` / `error`              |

---

## Privacy / PDPA

- **In-memory only** — ไม่เคย save image ลง disk, ไม่ cache OCR result
- **Logs ไม่มี PII** — log เฉพาะ `request_id`, method, path, status, duration_ms
  - มี defensive scrubber ปิด digit-run ≥ 9 หลัก เผื่อ regression ในอนาคต
- API key ผ่าน env var เท่านั้น
- TLS ให้ reverse proxy หน้า service จัดการ (nginx / traefik) — orc-service พูด HTTP plain
- Service ไม่เปิดให้ browser เรียกตรง — เฉพาะ NestJS ใน private network เท่านั้น

---

## Development

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt pytest httpx

pytest -v                              # unit tests (no engines required)
pytest tests/test_integration.py -v    # integration tests against fixtures in .test-fixtures/
```

### Local engine setup (สำหรับ run integration tests นอก docker)

| Engine | ทำไงบน Windows |
| ------ | -------------- |
| **Tesseract** (fastmrz / passport) | `winget install --id UB-Mannheim.TesseractOCR` |
| **mrz.traineddata** (fastmrz model) | `curl -L -o ~/tessdata/mrz.traineddata https://github.com/DoubangoTelecom/tesseractMRZ/raw/master/tessdata_best/mrz.traineddata` แล้ว conftest จะ pick up เอง |
| **PaddleOCR** (Thai ID) | ❌ ลง local บน Windows ติด Long Path limit (ไฟล์ใน `paddle/include/...` ยาวเกิน 260 ตัวอักษร) — ต้องเปิด long path support (admin + restart) หรือใช้ docker |

ทดสอบ Thai ID ครบจริงๆ แนะนำให้ build + run docker แล้ว curl ภาพเข้าโดยตรง:

```bash
docker compose build && docker compose up -d
curl -X POST http://localhost:8000/scan -H "X-API-Key: $(grep API_KEY .env | cut -d= -f2)" \
  -F type=thai_id -F image=@.test-fixtures/fake7.jpg
```

PaddleOCR + paddlepaddle รวม ~500MB ตอน install — แนะนำให้ใช้ docker เ ท่านั้น

โครงสร้างโค้ด: ดู `PRD.md §9`
