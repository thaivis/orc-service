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

## Kubernetes Deployment

Manifest แยกเป็นไฟล์ตาม resource อยู่ใต้ `k8s/`: `configmap.yaml`, `deployment.yaml`, `service.yaml`, `hpa.yaml`, `networkpolicy.yaml` ทุกไฟล์ตั้ง namespace เป็น `production` แล้ว แต่ไม่เก็บ production secret จริงไว้ใน git

### Bump image tag ผ่าน GitHub Action (วิธีปกติ)

Workflow `.github/workflows/deploy.yml` ทำ deploy image ตัวใหม่ให้ทั้งหมด: Actions → `deploy-production` → Run workflow → ใส่ `image_tag` (เช่น `sha-abc1234`, `v1.2.3`)

Workflow จะ verify ว่า tag มีอยู่จริงใน GHCR, แก้ image tag ใน `k8s/deployment.yaml`, apply แล้วรอ rollout — **commit กลับเข้า `main` เฉพาะตอน rollout สำเร็จเท่านั้น** ถ้า rollout fail จะไม่มี commit เกิดขึ้น เพื่อให้ `main` ตรงกับสิ่งที่ production รันอยู่เสมอ รันบน self-hosted runner (`self-hosted-thaivis`) ที่ตั้ง kubeconfig เข้าถึง `production` ไว้แล้ว

Workflow นี้ apply แค่ `k8s/deployment.yaml` เท่านั้น — ถ้าแก้ `configmap.yaml`, `service.yaml`, `hpa.yaml`, หรือ `networkpolicy.yaml` ต้อง apply เองด้วยมือตามขั้นตอนด้านล่าง

### Manual deploy / แก้ resource อื่นที่ไม่ใช่ image bump

### 1. Build และ push image

ตั้ง image tag ใน `k8s/deployment.yaml` ให้ตรงกับ image ที่ push แล้ว เช่น:

```yaml
image: ghcr.io/thaivis/orc-service:sha-06dbcba
```

ถ้า image เป็น private ต้องให้ Kubernetes มี pull secret ที่อ่าน `ghcr.io/thaivis/orc-service` ได้

### 2. สร้าง runtime API key secret

`API_KEY` ใช้สำหรับ NestJS เรียก orc-service อย่า commit key จริงลง git:

```bash
kubectl -n production create secret generic orc-service-secrets \
  --from-literal=API_KEY='<shared-api-key>' \
  --dry-run=client -o yaml | kubectl apply -f -
```

### 3. สร้าง GHCR pull secret

ใช้ GitHub account ที่มีสิทธิ์อ่าน package `ghcr.io/thaivis/orc-service` สร้าง GitHub Personal Access Token ที่มี scope `read:packages` ถ้า org `thaivis` เปิด SSO ต้อง authorize token กับ org ด้วย

อย่าพิมพ์ token ตรง ๆ ลง command line หรือแชต ถ้า token หลุด ให้ revoke ทันทีแล้วสร้างใหม่:

```bash
read -s GHCR_TOKEN
kubectl -n production create secret docker-registry orc-service-ghcr-pull \
  --docker-server=ghcr.io \
  --docker-username='<github-username-with-thaivis-access>' \
  --docker-password="$GHCR_TOKEN" \
  --docker-email='<email>' \
  --dry-run=client -o yaml | kubectl apply -f -
unset GHCR_TOKEN
```

Deployment ต้องอ้าง secret นี้:

```yaml
imagePullSecrets:
  - name: orc-service-ghcr-pull
```

### 4. ตรวจ diff แล้ว apply

ก่อน apply production ให้ดู diff ก่อน โดยเฉพาะ `Service` port และ `NetworkPolicy`:

```bash
kubectl diff -f k8s/
kubectl apply -f k8s/
```

ตาม rollout:

```bash
kubectl -n production rollout status deploy/orc-service
kubectl -n production get pods -l app=orc-service -o wide
```

ถ้า rollout ติด ให้ดู event ของ pod ใหม่:

```bash
kubectl -n production describe pod <pod-name>
```

สัญญาณที่พบบ่อย:

| Error | ความหมาย | วิธีแก้ |
| ----- | -------- | ------- |
| `not found` ตอน pull image | tag หรือ repository ผิด / image ยังไม่ได้ push | เช็ค image tag ใน manifest และ GHCR |
| `403 Forbidden` ตอน pull image | pull secret ไม่มีสิทธิ์อ่าน package | สร้าง `orc-service-ghcr-pull` ใหม่ด้วย PAT ที่มี `read:packages` และ org access |
| `Insufficient cpu` / `Insufficient memory` | cluster schedule pod ใหม่ไม่ได้ทันที | รอ autoscaler หรือเพิ่ม capacity / ลด requests อย่างระวัง |
| probe fail ที่ `/health` | container start แล้วแต่ app ยังไม่ ready | ดู logs และปรับ startup/readiness probe เฉพาะเมื่อ app ใช้เวลาบูตจริง |

หลัง rollout ผ่าน ทดสอบ health ผ่าน service:

```bash
kubectl -n production run orc-healthcheck --rm -it --restart=Never \
  --image=curlimages/curl -- \
  curl -fsS http://orc-service:8899/health
```

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
