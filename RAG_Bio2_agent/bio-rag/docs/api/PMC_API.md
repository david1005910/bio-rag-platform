# PMC (PubMed Central) API Documentation

PubMed Central에서 Open Access 논문의 PDF 정보를 조회하고 다운로드하는 API입니다.

## Base URL

```
/api/v1
```

---

## Endpoints

### 1. 단일 논문 PDF 정보 조회

논문의 PDF 이용 가능 여부와 다운로드 URL을 조회합니다.

**Endpoint**
```
GET /papers/{pmid}/pdf-info
```

**Path Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| pmid | string | Yes | PubMed ID |

**Response**

```json
{
  "pmid": "32015507",
  "pmcid": "PMC7095418",
  "has_pdf": true,
  "pdf_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7095418/pdf/41586_2020_Article_2012.pdf",
  "is_open_access": true
}
```

**Response Fields**

| Field | Type | Description |
|-------|------|-------------|
| pmid | string | PubMed ID |
| pmcid | string \| null | PubMed Central ID (없으면 null) |
| has_pdf | boolean | PDF 이용 가능 여부 |
| pdf_url | string \| null | PDF 다운로드 URL (없으면 null) |
| is_open_access | boolean | Open Access 여부 |

**Example Request**

```bash
curl -X GET "http://localhost:8000/api/v1/papers/32015507/pdf-info"
```

**Example Response (PDF 이용 가능)**

```json
{
  "pmid": "32015507",
  "pmcid": "PMC7095418",
  "has_pdf": true,
  "pdf_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7095418/pdf/41586_2020_Article_2012.pdf",
  "is_open_access": true
}
```

**Example Response (PDF 미제공)**

```json
{
  "pmid": "12345678",
  "pmcid": null,
  "has_pdf": false,
  "pdf_url": null,
  "is_open_access": false
}
```

---

### 2. 여러 논문 PDF 정보 일괄 조회

여러 논문의 PDF 정보를 한 번에 조회합니다.

**Endpoint**
```
POST /papers/pdf-info-batch
```

**Request Body**

```json
{
  "pmids": ["32015507", "35294395", "12345678"]
}
```

**Request Fields**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| pmids | string[] | Yes | PubMed ID 목록 |

**Response**

```json
{
  "papers": [
    {
      "pmid": "32015507",
      "pmcid": "PMC7095418",
      "has_pdf": true,
      "pdf_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7095418/pdf/...",
      "is_open_access": true
    },
    {
      "pmid": "35294395",
      "pmcid": "PMC8885344",
      "has_pdf": true,
      "pdf_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8885344/pdf/...",
      "is_open_access": true
    },
    {
      "pmid": "12345678",
      "pmcid": null,
      "has_pdf": false,
      "pdf_url": null,
      "is_open_access": false
    }
  ]
}
```

**Example Request**

```bash
curl -X POST "http://localhost:8000/api/v1/papers/pdf-info-batch" \
  -H "Content-Type: application/json" \
  -d '{"pmids": ["32015507", "35294395", "12345678"]}'
```

---

### 3. PDF 다운로드

논문 PDF를 직접 다운로드합니다.

**Endpoint**
```
GET /papers/{pmid}/pdf
```

**Path Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| pmid | string | Yes | PubMed ID |

**Response**

- **Success (200)**: PDF 파일 (application/pdf)
- **Not Found (404)**: PDF 미제공

**Response Headers (Success)**

```
Content-Type: application/pdf
Content-Disposition: attachment; filename=32015507_PMC7095418.pdf
```

**Example Request**

```bash
curl -X GET "http://localhost:8000/api/v1/papers/32015507/pdf" \
  -o paper.pdf
```

**Error Response (404)**

```json
{
  "detail": "PDF not available for PMID 12345678"
}
```

---

## Error Responses

### Common Error Codes

| Status Code | Description |
|-------------|-------------|
| 200 | 성공 |
| 404 | PDF를 찾을 수 없음 |
| 500 | 서버 내부 오류 |

### Error Response Format

```json
{
  "detail": "Error message here"
}
```

---

## Usage Notes

### PDF 이용 가능 조건

1. **PMC 등록**: 논문이 PubMed Central에 등록되어 있어야 함
2. **Open Access**: Open Access로 공개된 논문만 PDF 다운로드 가능
3. **PMCID 필요**: PMID → PMCID 변환이 성공해야 함

### 권장 사용 패턴

```javascript
// 1. 먼저 PDF 정보 확인
const pdfInfo = await api.get(`/papers/${pmid}/pdf-info`);

// 2. PDF 이용 가능 시 다운로드
if (pdfInfo.has_pdf && pdfInfo.pdf_url) {
  // 방법 1: URL 직접 열기 (권장)
  window.open(pdfInfo.pdf_url, '_blank');

  // 방법 2: API를 통한 다운로드
  const response = await api.get(`/papers/${pmid}/pdf`, {
    responseType: 'blob'
  });
}
```

### Rate Limiting

- NCBI API는 초당 3회 요청 제한
- 대량 요청 시 `pdf-info-batch` 엔드포인트 사용 권장

---

## Frontend Integration

### TypeScript Interface

```typescript
interface PDFInfo {
  pmid: string;
  pmcid?: string;
  hasPdf: boolean;
  pdfUrl?: string;
  isOpenAccess: boolean;
}
```

### React Example

```tsx
const [pdfInfo, setPdfInfo] = useState<PDFInfo | null>(null);
const [isLoading, setIsLoading] = useState(false);

const handlePdfClick = async () => {
  if (!pdfInfo) {
    setIsLoading(true);
    const info = await searchApi.getPdfInfo(pmid);
    setPdfInfo(info);

    if (info.hasPdf && info.pdfUrl) {
      window.open(info.pdfUrl, '_blank');
    }
    setIsLoading(false);
  } else if (pdfInfo.hasPdf && pdfInfo.pdfUrl) {
    window.open(pdfInfo.pdfUrl, '_blank');
  }
};
```

---

## Related Services

| Service | Description |
|---------|-------------|
| `PMCService` | PMC API 클라이언트 (`src/services/pmc.py`) |
| `searchApi` | 프론트엔드 API 클라이언트 (`src/services/api.ts`) |

---

## References

- [PubMed Central](https://www.ncbi.nlm.nih.gov/pmc/)
- [NCBI ID Converter API](https://www.ncbi.nlm.nih.gov/pmc/tools/id-converter-api/)
- [PMC Open Access Subset](https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/)
