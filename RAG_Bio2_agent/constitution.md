# Bio-RAG 프로젝트 Constitution (프로젝트 헌법)

> 이 문서는 Bio-RAG 프로젝트의 모든 개발 결정을 안내하는 핵심 원칙과 가이드라인을 정의합니다.
> 모든 specification, plan, task는 이 constitution을 준수해야 합니다.

---

## 1. 프로젝트 비전 및 미션

### 1.1 비전
"모든 바이오 연구자가 최신 연구 동향을 실시간으로 파악하고, AI와 대화하듯 논문을 분석할 수 있는 세상"

### 1.2 미션
- 논문 검색 및 분석 시간 70% 단축
- 연구 트렌드 파악 시간 80% 감소
- 관련 논문 발견율 200% 향상

### 1.3 핵심 가치
1. **정확성 (Accuracy)**: RAG 기반 답변의 정확도 95% 이상 유지
2. **속도 (Speed)**: 평균 응답 시간 2초 이내
3. **신뢰성 (Reliability)**: 99.5% 가용성, 할루시네이션 검증 필수
4. **사용성 (Usability)**: 5분 내 핵심 기능 학습 가능

---

## 2. 기술 원칙

### 2.1 아키텍처 원칙

#### 2.1.1 레이어 분리
```
반드시 다음 4개 레이어로 구조화할 것:
1. Client Layer (프론트엔드)
2. Application Layer (API 서비스)
3. Data Processing Layer (데이터 처리)
4. Storage Layer (데이터 저장)
```

**이유**: 관심사 분리를 통한 유지보수성 향상, 독립적 배포 가능

#### 2.1.2 마이크로서비스 지향
```
핵심 서비스는 독립적으로 분리:
- Search Service: 논문 검색 담당
- RAG Service: AI 질의응답 담당
- Analytics Service: 트렌드 분석 담당
```

**이유**: 서비스별 독립 확장, 장애 격리

#### 2.1.3 비동기 처리
```
CPU 집약적 작업은 반드시 비동기 처리:
- 논문 크롤링
- 임베딩 생성
- 배치 프로세싱
```

**이유**: 사용자 응답성 유지, 리소스 효율화

### 2.2 코드 품질 원칙

#### 2.2.1 타입 안전성
```python
# ✅ 올바른 예시
from pydantic import BaseModel

class PaperMetadata(BaseModel):
    pmid: str
    title: str
    abstract: str
    authors: List[str]

# ❌ 피해야 할 예시
def get_paper(data):  # 타입 힌트 없음
    return data['title']
```

**규칙**:
- Python: Pydantic 모델 필수 사용
- TypeScript: strict mode 활성화
- 모든 함수에 타입 힌트 필수

#### 2.2.2 에러 처리
```python
# ✅ 올바른 예시
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def api_call_with_retry(url: str) -> dict:
    try:
        response = await httpx.get(url)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        logger.error(f"API call failed: {e}")
        raise PubMedAPIError(f"Failed to fetch: {url}")
```

**규칙**:
- 외부 API 호출 시 재시도 로직 필수
- 사용자 친화적 에러 메시지
- 모든 예외는 로깅 필수

#### 2.2.3 테스트 커버리지
```
최소 테스트 커버리지 요구사항:
- Unit Test: 80% 이상
- Integration Test: 핵심 플로우 100%
- E2E Test: 주요 사용자 시나리오 커버
```

### 2.3 성능 원칙

#### 2.3.1 응답 시간 제약
| 작업 | 최대 시간 |
|------|----------|
| AI 챗봇 응답 | 5초 |
| 논문 검색 | 1초 |
| 페이지 로드 | 1초 |
| 유사 논문 추천 | 2초 |

#### 2.3.2 캐싱 전략
```
필수 캐싱 적용 대상:
- LLM API 응답 (Redis, TTL: 7일)
- 검색 결과 (Redis, TTL: 1시간)
- 임베딩 결과 (영구 저장)
- 정적 메타데이터 (메모리 캐시)
```

#### 2.3.3 확장성
```
수평 확장 가능 설계:
- Stateless 서비스 설계
- 데이터베이스 Connection Pooling
- 벡터 DB 샤딩 지원
```

---

## 3. 데이터 원칙

### 3.1 데이터 수집
```
PubMed API 사용 규칙:
- Rate Limit: 10 req/sec (API 키 사용 시)
- 일일 배치 크롤링: UTC 02:00
- 필수 수집 필드: PMID, Title, Abstract, Authors, Journal, Date, Keywords, MeSH Terms
```

### 3.2 데이터 품질
```
텍스트 전처리 필수 단계:
1. 특수문자 정제
2. 참조 번호 제거 ([1], [2] 등)
3. 연속 공백 정규화
4. 표/그림 캡션 제거 (옵션)
```

### 3.3 임베딩 전략
```
모델: PubMedBERT (microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract)
차원: 768
청킹: 
- 기본: 512 토큰, 50 토큰 오버랩
- 논문별: 섹션 기반 청킹 (abstract, methods, results 등)
```

---

## 4. AI/RAG 원칙

### 4.1 프롬프트 엔지니어링
```
필수 포함 요소:
1. 역할 정의 (바이오메디컬 연구 전문가)
2. 컨텍스트 기반 응답 제약
3. PMID 인용 형식 지정
4. 할루시네이션 방지 지침
```

**프롬프트 템플릿 예시**:
```
You are an expert biomedical researcher assistant.

IMPORTANT RULES:
1. Only use information from the provided context
2. Cite sources using [PMID: xxxxx] format
3. If the context doesn't contain enough information, say "I cannot find sufficient information in the provided papers"
4. Do not make assumptions or add information not present in the context
```

### 4.2 검색 증강 생성 (RAG) 파이프라인
```
필수 단계:
1. Query Embedding (질문 → 벡터)
2. Vector Search (Top-K 검색, K >= 5)
3. Re-ranking (선택적, CrossEncoder 사용)
4. Context Building (검색 결과 조합)
5. LLM Generation (프롬프트 + 컨텍스트)
6. Response Validation (할루시네이션 체크)
```

### 4.3 할루시네이션 검증
```
필수 검증 항목:
1. 인용된 PMID가 검색 결과에 존재하는지 확인
2. 응답 내용이 소스 텍스트와 일치하는지 확인
3. 신뢰도 점수 계산 및 표시
```

---

## 5. 보안 및 개인정보 원칙

### 5.1 인증/인가
```
필수 적용:
- OAuth 2.0 / JWT 토큰 기반 인증
- API 키는 환경 변수 또는 Vault 관리
- 세션 만료: 24시간
```

### 5.2 데이터 암호화
```
암호화 규격:
- 전송 중: TLS 1.3
- 저장 시: AES-256
- 해시: bcrypt (패스워드)
```

### 5.3 규정 준수
```
준수 대상:
- GDPR (유럽 개인정보보호)
- PIPA (한국 개인정보보호법)
- PubMed 저작권 (Fair Use)
```

---

## 6. 운영 원칙

### 6.1 가용성
```
SLA 목표:
- Uptime: 99.5% (월 3.6시간 이내 다운타임)
- RTO: 1시간
- RPO: 1시간
```

### 6.2 모니터링
```
필수 메트릭:
- Request Volume (요청량)
- Latency Percentiles (p50, p95, p99)
- Error Rate (%)
- LLM API Costs ($)
- Cache Hit Rate (%)
```

### 6.3 로깅
```
로깅 규칙:
- 모든 API 요청/응답 로깅
- 사용자 검색 쿼리 익명화 로깅
- 에러 발생 시 스택 트레이스 포함
- 로그 보관: 30일
```

---

## 7. 개발 프로세스 원칙

### 7.1 브랜치 전략
```
Git Flow:
- main: 프로덕션 배포
- develop: 개발 통합
- feature/*: 기능 개발
- hotfix/*: 긴급 수정
```

### 7.2 코드 리뷰
```
필수 조건:
- 모든 PR은 최소 1명 리뷰 필요
- CI 파이프라인 통과 필수
- 테스트 커버리지 유지
```

### 7.3 배포
```
배포 전략:
- Blue-Green Deployment (무중단 배포)
- Feature Flag 활용
- Rollback 계획 필수
```

---

## 8. 기술 스택 제약

### 8.1 프론트엔드
```
필수:
- React.js 18.2+
- TypeScript 5.0+
- TailwindCSS 3.3+
- React Query 4.0+
- Recharts 2.5+ (시각화)
```

### 8.2 백엔드
```
필수:
- Python 3.11+
- FastAPI 0.104+
- Pydantic 2.0+
- Celery 5.3+ (비동기 작업)
- Redis 7.0+ (캐싱/큐)
```

### 8.3 AI/ML
```
필수:
- LangChain 0.1+
- Hugging Face Transformers 4.35+
- PubMedBERT (임베딩)
- OpenAI API GPT-4 (LLM)
```

### 8.4 데이터베이스
```
필수:
- PostgreSQL 15+ (메타데이터)
- Qdrant 또는 ChromaDB (벡터 DB)
- FAISS (대규모 벡터 검색)
```

### 8.5 인프라
```
AWS 기반:
- ECS/EKS (컨테이너 오케스트레이션)
- RDS PostgreSQL (관계형 DB)
- ElastiCache Redis (캐싱)
- S3 (오브젝트 스토리지)
- CloudFront (CDN)
```

---

## 9. 의사결정 프레임워크

### 기술 결정 시 질문
1. 이 결정이 핵심 가치(정확성, 속도, 신뢰성, 사용성)에 부합하는가?
2. 팀이 유지보수할 수 있는가?
3. 확장 가능한가?
4. 장기적인 비용은 어떠한가?
5. Constitution의 원칙을 위반하지 않는가?

### 트레이드오프 우선순위
```
1. 정확성 > 속도 (연구자에게 잘못된 정보는 치명적)
2. 신뢰성 > 기능 추가 (안정적인 서비스가 우선)
3. 사용자 경험 > 기술적 우아함 (실용성 중시)
4. 보안 > 편의성 (데이터 보호 필수)
```

---

## 10. Glossary (용어집)

| 용어 | 정의 |
|------|------|
| RAG | Retrieval-Augmented Generation - 검색 증강 생성 |
| Embedding | 텍스트를 고차원 벡터로 변환 |
| PMID | PubMed Unique Identifier |
| MeSH | Medical Subject Headings - 의학 주제어 |
| Chunking | 텍스트를 작은 단위로 분할 |
| Hallucination | LLM이 사실과 다른 정보를 생성하는 현상 |
| BioBERT | 생명과학 도메인에 특화된 BERT 모델 |
| Vector DB | 벡터 유사도 검색을 위한 데이터베이스 |

---

## 변경 이력

| 버전 | 날짜 | 변경 내용 |
|------|------|----------|
| 1.0 | 2024.12 | 초기 Constitution 작성 |

---

> ⚠️ **중요**: 이 Constitution은 프로젝트의 모든 결정에 우선합니다.
> 변경이 필요한 경우, 팀 전체의 합의를 거쳐 문서화해야 합니다.
