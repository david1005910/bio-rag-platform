import { useState, useEffect, useCallback, useRef } from 'react'
import { Play, Pause, RotateCcw, ChevronRight, Volume2, VolumeX } from 'lucide-react'
import { speakText, stopSpeech, loadVoices } from '@/utils/speech'

interface PipelineStep {
  id: number
  name: string
  nameKo: string
  description: string
  color: string
  icon: string
  details: string[]
  narration: string  // 음성 설명 텍스트
}

const PIPELINE_STEPS: PipelineStep[] = [
  {
    id: 1,
    name: 'Data Collection',
    nameKo: '데이터 수집',
    description: 'PubMed API에서 논문 메타데이터 수집',
    color: '#3B82F6',
    icon: '📥',
    details: [
      'PubMed E-utilities API 호출',
      '논문 제목, 초록, 저자 추출',
      'Rate Limit: 10 req/sec',
    ],
    narration: '첫 번째 단계, 데이터 수집입니다. PubMed API를 호출하여 바이오메디컬 논문의 제목, 초록, 저자 정보를 수집합니다.',
  },
  {
    id: 2,
    name: 'Text Preprocessing',
    nameKo: '텍스트 전처리',
    description: '텍스트 정제 및 청킹',
    color: '#10B981',
    icon: '🔧',
    details: ['특수문자 제거', '참조번호 정규화', '512 토큰 단위 청킹'],
    narration: '두 번째 단계, 텍스트 전처리입니다. 수집된 텍스트에서 특수문자를 제거하고, 512 토큰 단위로 청킹합니다.',
  },
  {
    id: 3,
    name: 'Embedding Generation',
    nameKo: '임베딩 생성',
    description: 'OpenAI API로 벡터 임베딩 생성',
    color: '#8B5CF6',
    icon: '🧮',
    details: ['text-embedding-3-small 모델', '1536 차원 벡터', '배치 처리 (100개씩)'],
    narration: '세 번째 단계, 임베딩 생성입니다. OpenAI의 임베딩 모델을 사용하여 텍스트를 1536차원 벡터로 변환합니다.',
  },
  {
    id: 4,
    name: 'Vector Storage',
    nameKo: '벡터 저장',
    description: 'Qdrant에 벡터 인덱싱',
    color: '#F59E0B',
    icon: '💾',
    details: ['Qdrant Vector DB', 'HNSW 인덱스', '메타데이터 저장'],
    narration: '네 번째 단계, 벡터 저장입니다. 생성된 벡터를 Qdrant 벡터 데이터베이스에 HNSW 인덱스로 저장합니다.',
  },
  {
    id: 5,
    name: 'Query Processing',
    nameKo: '쿼리 처리',
    description: '사용자 질문 처리 및 임베딩',
    color: '#EC4899',
    icon: '❓',
    details: ['한글 → 영어 번역', '쿼리 임베딩 생성', '검색 파라미터 설정'],
    narration: '다섯 번째 단계, 쿼리 처리입니다. 사용자의 질문을 영어로 번역하고, 검색을 위한 임베딩을 생성합니다.',
  },
  {
    id: 6,
    name: 'Memory Search',
    nameKo: '메모리 검색',
    description: 'SQLite에서 유사한 과거 Q&A 검색',
    color: '#DB2777',
    icon: '🧠',
    details: ['FTS5 전문 검색', 'BM25 유사도 랭킹', '과거 대화 컨텍스트 추출'],
    narration: '여섯 번째 단계, 메모리 검색입니다. SQLite 데이터베이스에서 유사한 과거 질문과 답변을 검색합니다.',
  },
  {
    id: 7,
    name: 'Hybrid Search',
    nameKo: '하이브리드 검색',
    description: 'Dense + Sparse 검색 융합',
    color: '#06B6D4',
    icon: '🔍',
    details: ['Dense: 의미 유사도 (70%)', 'Sparse: 키워드 매칭 (30%)', 'Score Fusion'],
    narration: '일곱 번째 단계, 하이브리드 검색입니다. 의미 기반 Dense 검색과 키워드 기반 Sparse 검색을 70대 30 비율로 융합합니다.',
  },
  {
    id: 8,
    name: 'Graph Search',
    nameKo: 'GraphDB 검색',
    description: 'Neo4j 그래프 관계 기반 검색',
    color: '#A855F7',
    icon: '🕸️',
    details: ['Neo4j 인용 네트워크 탐색', '저자/키워드 관계 분석', 'RRF 점수 융합'],
    narration: '여덟 번째 단계, GraphDB 검색입니다. Neo4j 그래프 데이터베이스에서 논문 간 인용 관계와 저자, 키워드 네트워크를 탐색합니다.',
  },
  {
    id: 9,
    name: 'Reranking',
    nameKo: '리랭킹',
    description: 'Cross-Encoder로 검색 결과 재정렬',
    color: '#F97316',
    icon: '🎯',
    details: ['Cross-Encoder 모델', '쿼리-문서 관련성 재평가', 'Top-K 재정렬'],
    narration: '아홉 번째 단계, 리랭킹입니다. Cross-Encoder 모델로 검색 결과의 관련성을 재평가하여 순위를 조정합니다.',
  },
  {
    id: 10,
    name: 'Context Building',
    nameKo: '컨텍스트 구성',
    description: '검색 결과 + 메모리로 프롬프트 구성',
    color: '#EF4444',
    icon: '📋',
    details: ['Top-K 문서 선택', '메모리 컨텍스트 병합', '프롬프트 템플릿 적용'],
    narration: '열 번째 단계, 컨텍스트 구성입니다. 검색된 논문과 메모리 정보를 결합하여 AI 프롬프트를 구성합니다.',
  },
  {
    id: 11,
    name: 'LLM Generation',
    nameKo: 'LLM 응답 생성',
    description: 'GPT-4로 답변 생성',
    color: '#22C55E',
    icon: '🤖',
    details: ['GPT-4 API 호출', '컨텍스트 기반 응답', '출처 인용 포함'],
    narration: '열한 번째 단계, LLM 응답 생성입니다. GPT-4 모델이 구성된 컨텍스트를 바탕으로 답변을 생성합니다.',
  },
  {
    id: 12,
    name: 'Memory Save',
    nameKo: '메모리 저장',
    description: 'Q&A를 SQLite에 저장',
    color: '#7C3AED',
    icon: '💿',
    details: ['질문-답변 쌍 저장', '쿼리 해시 인덱싱', 'FTS 트리거 업데이트'],
    narration: '마지막 단계, 메모리 저장입니다. 질문과 답변을 SQLite 데이터베이스에 저장하여 다음 질문에 참조합니다.',
  },
]

export default function PipelineAnimation() {
  const [currentStep, setCurrentStep] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [showDetails, setShowDetails] = useState(true)
  const [voiceEnabled, setVoiceEnabled] = useState(false)
  const [isSpeaking, setIsSpeaking] = useState(false)
  const isPlayingRef = useRef(false)  // 재생 상태 추적용

  // 음성 목록 미리 로드 (한번만)
  useEffect(() => {
    loadVoices()
  }, [])

  // isPlaying 상태를 ref에 동기화
  useEffect(() => {
    isPlayingRef.current = isPlaying
  }, [isPlaying])

  // 다음 단계로 이동하는 함수
  const goToNextStep = useCallback(() => {
    setCurrentStep((prev) => {
      if (prev >= PIPELINE_STEPS.length - 1) {
        setIsPlaying(false)
        return prev
      }
      return prev + 1
    })
  }, [])

  // 음성 합성 함수 (콜백으로 다음 단계 이동)
  const speak = useCallback((text: string, onComplete?: () => void) => {
    if (!voiceEnabled) {
      // 음성 비활성화 시 바로 콜백 실행
      onComplete?.()
      return
    }

    setIsSpeaking(true)

    speakText(text, {
      lang: 'ko',
      rate: 1.0,
      pitch: 1.1,
      onStart: () => setIsSpeaking(true),
      onEnd: () => {
        setIsSpeaking(false)
        // 음성 종료 후 재생 중이면 다음 단계로 이동
        if (isPlayingRef.current && onComplete) {
          setTimeout(onComplete, 500)
        }
      },
      onError: () => {
        setIsSpeaking(false)
        onComplete?.()
      }
    })
  }, [voiceEnabled])

  // 음성 중지 함수
  const stopSpeaking = useCallback(() => {
    stopSpeech()
    setIsSpeaking(false)
  }, [])

  // 단계 변경 시 음성 재생 (음성 활성화 + 재생 중일 때)
  useEffect(() => {
    if (voiceEnabled && isPlaying) {
      speak(PIPELINE_STEPS[currentStep].narration, goToNextStep)
    }
  }, [currentStep, voiceEnabled, isPlaying, speak, goToNextStep])

  // 음성 기능 토글 시 현재 단계 설명
  useEffect(() => {
    if (voiceEnabled && !isPlaying) {
      speak(PIPELINE_STEPS[currentStep].narration)
    } else if (!voiceEnabled) {
      stopSpeaking()
    }
  }, [voiceEnabled])

  // 컴포넌트 언마운트 시 음성 중지
  useEffect(() => {
    return () => {
      stopSpeaking()
    }
  }, [stopSpeaking])

  // 애니메이션 (음성 비활성화 시에만 interval 사용)
  useEffect(() => {
    let interval: ReturnType<typeof setInterval> | null = null

    if (isPlaying && !voiceEnabled) {
      // 음성 비활성화 시: 2초 간격으로 자동 진행
      interval = setInterval(() => {
        setCurrentStep((prev) => {
          if (prev >= PIPELINE_STEPS.length - 1) {
            setIsPlaying(false)
            return prev
          }
          return prev + 1
        })
      }, 2000)
    }
    // 음성 활성화 시: speak 함수의 onend 콜백에서 다음 단계로 이동

    return () => {
      if (interval) clearInterval(interval)
    }
  }, [isPlaying, voiceEnabled])

  const handleReset = () => {
    setCurrentStep(0)
    setIsPlaying(false)
    stopSpeaking()
  }

  const handleStepClick = (index: number) => {
    setCurrentStep(index)
    setIsPlaying(false)
    if (voiceEnabled) {
      speak(PIPELINE_STEPS[index].narration)
    }
  }

  const toggleVoice = () => {
    if (voiceEnabled) {
      stopSpeaking()
    }
    setVoiceEnabled(!voiceEnabled)
  }

  return (
    <div className="glossy-panel p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-semibold text-white flex items-center gap-2">
            <span className="text-2xl">⚡</span>
            RAG 파이프라인
          </h2>
          <p className="text-white/60 text-sm mt-1">
            데이터 수집부터 AI 응답까지의 처리 과정
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className={`p-2 rounded-lg transition-all ${
              isPlaying
                ? 'bg-orange-500/20 text-orange-400 border border-orange-400/30'
                : 'bg-green-500/20 text-green-400 border border-green-400/30'
            }`}
          >
            {isPlaying ? <Pause size={20} /> : <Play size={20} />}
          </button>
          <button
            onClick={handleReset}
            className="p-2 rounded-lg bg-white/10 text-white/70 border border-white/20 hover:bg-white/20 transition-all"
          >
            <RotateCcw size={20} />
          </button>
          <button
            onClick={() => setShowDetails(!showDetails)}
            className={`px-3 py-2 rounded-lg text-sm transition-all ${
              showDetails
                ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-400/30'
                : 'bg-white/10 text-white/70 border border-white/20'
            }`}
          >
            상세 {showDetails ? 'ON' : 'OFF'}
          </button>
          <button
            onClick={toggleVoice}
            className={`p-2 rounded-lg transition-all flex items-center gap-2 ${
              voiceEnabled
                ? 'bg-pink-500/20 text-pink-400 border border-pink-400/30'
                : 'bg-white/10 text-white/70 border border-white/20'
            } ${isSpeaking ? 'animate-pulse' : ''}`}
            title={voiceEnabled ? '음성 설명 끄기' : '음성 설명 켜기'}
          >
            {voiceEnabled ? <Volume2 size={20} /> : <VolumeX size={20} />}
            <span className="text-sm hidden sm:inline">음성</span>
          </button>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="mb-6">
        <div className="flex justify-between text-xs text-white/50 mb-2">
          <span>시작</span>
          <span>
            {currentStep + 1} / {PIPELINE_STEPS.length}
          </span>
          <span>완료</span>
        </div>
        <div className="h-2 bg-white/10 rounded-full overflow-hidden">
          <div
            className="h-full transition-all duration-500 ease-out rounded-full"
            style={{
              width: `${((currentStep + 1) / PIPELINE_STEPS.length) * 100}%`,
              background: `linear-gradient(90deg, ${PIPELINE_STEPS[0].color}, ${PIPELINE_STEPS[currentStep].color})`,
            }}
          />
        </div>
      </div>

      {/* Pipeline Steps - Desktop: 2 rows, Mobile: Single column */}
      <div className="hidden lg:block">
        {/* Top Row (Steps 1-6) */}
        <div className="flex items-center justify-between mb-4">
          {PIPELINE_STEPS.slice(0, 6).map((step, index) => (
            <div key={step.id} className="flex items-center">
              <StepBox
                step={step}
                isActive={currentStep === index}
                isPast={currentStep > index}
                onClick={() => handleStepClick(index)}
                showDetails={showDetails && currentStep === index}
              />
              {index < 5 && (
                <div className="mx-1">
                  <ChevronRight
                    size={20}
                    className={`transition-all duration-300 ${
                      currentStep > index ? 'text-cyan-400' : 'text-white/20'
                    }`}
                  />
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Connector between rows */}
        <div className="flex justify-end pr-[50px] mb-4">
          <div
            className={`w-1 h-8 rounded-full transition-all duration-300 ${
              currentStep >= 6 ? 'bg-cyan-400' : 'bg-white/20'
            }`}
          />
        </div>

        {/* Bottom Row (Steps 7-12) - Reversed order for flow */}
        <div className="flex items-center justify-between flex-row-reverse">
          {PIPELINE_STEPS.slice(6)
            .reverse()
            .map((step, revIndex) => {
              const index = 11 - revIndex
              return (
                <div key={step.id} className="flex items-center">
                  {revIndex < 5 && (
                    <div className="mx-1">
                      <ChevronRight
                        size={20}
                        className={`rotate-180 transition-all duration-300 ${
                          currentStep > index ? 'text-cyan-400' : 'text-white/20'
                        }`}
                      />
                    </div>
                  )}
                  <StepBox
                    step={step}
                    isActive={currentStep === index}
                    isPast={currentStep > index}
                    onClick={() => handleStepClick(index)}
                    showDetails={showDetails && currentStep === index}
                  />
                </div>
              )
            })}
        </div>
      </div>

      {/* Mobile View - Vertical */}
      <div className="lg:hidden space-y-3">
        {PIPELINE_STEPS.map((step, index) => (
          <div key={step.id} className="flex items-start gap-3">
            <div className="flex flex-col items-center">
              <div
                className={`w-10 h-10 rounded-full flex items-center justify-center text-lg transition-all duration-300 ${
                  currentStep === index
                    ? 'ring-2 ring-offset-2 ring-offset-slate-900 scale-110 ring-cyan-400'
                    : currentStep > index
                      ? 'opacity-70'
                      : 'opacity-40'
                }`}
                style={{
                  backgroundColor: step.color,
                }}
              >
                {step.icon}
              </div>
              {index < PIPELINE_STEPS.length - 1 && (
                <div
                  className={`w-0.5 h-8 mt-2 transition-all duration-300 ${
                    currentStep > index ? 'bg-cyan-400' : 'bg-white/20'
                  }`}
                />
              )}
            </div>
            <div
              className={`flex-1 pb-4 transition-all duration-300 cursor-pointer ${
                currentStep === index ? 'opacity-100' : 'opacity-50'
              }`}
              onClick={() => handleStepClick(index)}
            >
              <div className="flex items-center gap-2">
                <span className="font-semibold text-white">{step.nameKo}</span>
                <span className="text-xs text-white/50">Step {step.id}</span>
              </div>
              <p className="text-sm text-white/70 mt-1">{step.description}</p>
              {showDetails && currentStep === index && (
                <div className="mt-2 space-y-1">
                  {step.details.map((detail, i) => (
                    <div key={i} className="flex items-center gap-2 text-xs text-white/60">
                      <div
                        className="w-1.5 h-1.5 rounded-full"
                        style={{ backgroundColor: step.color }}
                      />
                      {detail}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Current Step Details Panel (Desktop) */}
      {showDetails && (
        <div className="hidden lg:block mt-6 p-4 rounded-xl border border-white/10 bg-white/5">
          <div className="flex items-start gap-4">
            <div
              className="w-14 h-14 rounded-xl flex items-center justify-center text-2xl shrink-0"
              style={{ backgroundColor: PIPELINE_STEPS[currentStep].color }}
            >
              {PIPELINE_STEPS[currentStep].icon}
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-1">
                <h3 className="font-semibold text-white text-lg">
                  {PIPELINE_STEPS[currentStep].nameKo}
                </h3>
                <span className="text-xs px-2 py-0.5 rounded-full bg-white/10 text-white/60">
                  Step {PIPELINE_STEPS[currentStep].id}
                </span>
              </div>
              <p className="text-white/80 text-sm mb-3">{PIPELINE_STEPS[currentStep].description}</p>
              <div className="flex flex-wrap gap-2">
                {PIPELINE_STEPS[currentStep].details.map((detail, i) => (
                  <span
                    key={i}
                    className="px-3 py-1 rounded-full text-xs font-medium"
                    style={{
                      backgroundColor: `${PIPELINE_STEPS[currentStep].color}20`,
                      color: PIPELINE_STEPS[currentStep].color,
                      border: `1px solid ${PIPELINE_STEPS[currentStep].color}40`,
                    }}
                  >
                    {detail}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Code Example */}
      <div className="mt-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-white/60">코드 예시</span>
          <span className="text-xs text-white/40">{PIPELINE_STEPS[currentStep].name}</span>
        </div>
        <CodeExample step={PIPELINE_STEPS[currentStep]} />
      </div>
    </div>
  )
}

function StepBox({
  step,
  isActive,
  isPast,
  onClick,
}: {
  step: PipelineStep
  isActive: boolean
  isPast: boolean
  onClick: () => void
  showDetails?: boolean
}) {
  return (
    <div
      onClick={onClick}
      className={`relative cursor-pointer transition-all duration-300 ${
        isActive ? 'scale-105 z-10' : isPast ? 'opacity-70' : 'opacity-40'
      }`}
    >
      <div
        className={`w-28 h-24 rounded-xl p-2 flex flex-col items-center justify-center transition-all duration-300 ${
          isActive ? 'ring-2 ring-offset-2 ring-offset-slate-900 ring-cyan-400' : ''
        }`}
        style={{
          backgroundColor: isActive ? step.color : `${step.color}30`,
          borderColor: step.color,
          border: `2px solid ${step.color}`,
        }}
      >
        <span className="text-2xl mb-1">{step.icon}</span>
        <span className="text-white text-xs font-semibold text-center">{step.nameKo}</span>
        <span className="text-white/60 text-[10px]">Step {step.id}</span>
      </div>

      {/* Pulse animation for active step */}
      {isActive && (
        <div
          className="absolute inset-0 rounded-xl animate-ping opacity-30"
          style={{ backgroundColor: step.color }}
        />
      )}
    </div>
  )
}

function CodeExample({ step }: { step: PipelineStep }) {
  const codeExamples: Record<number, string> = {
    1: `# PubMed API 데이터 수집
async def fetch_papers(query: str, max_results: int = 100):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {"db": "pubmed", "term": query, "retmax": max_results}
    response = await httpx.get(url, params=params)
    return parse_pubmed_response(response.json())`,

    2: `# 텍스트 전처리 및 청킹
def preprocess_text(text: str) -> list[str]:
    # 특수문자 제거 및 정규화
    cleaned = re.sub(r'\\[\\d+\\]', '', text)  # 참조번호 제거
    cleaned = re.sub(r'[^\\w\\s.-]', '', cleaned)

    # 512 토큰 단위로 청킹
    chunks = split_into_chunks(cleaned, max_tokens=512)
    return chunks`,

    3: `# OpenAI 임베딩 생성
async def generate_embeddings(texts: list[str]) -> list[list[float]]:
    response = await openai.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
        dimensions=1536
    )
    return [item.embedding for item in response.data]`,

    4: `# Qdrant 벡터 저장
async def store_vectors(embeddings: list, metadata: list):
    points = [
        PointStruct(
            id=str(uuid4()),
            vector=emb,
            payload=meta
        )
        for emb, meta in zip(embeddings, metadata)
    ]
    await qdrant.upsert(collection_name="papers", points=points)`,

    5: `# 쿼리 처리
async def process_query(question: str) -> dict:
    # 한글 → 영어 번역 (필요시)
    if detect_language(question) == "ko":
        question = await translate(question, "ko", "en")

    # 쿼리 임베딩 생성
    query_embedding = await generate_embedding(question)
    return {"embedding": query_embedding, "original": question}`,

    6: `# 메모리 검색 (SQLite FTS5)
def search_similar_conversations(query: str, limit: int = 3):
    # 불용어 제거 후 키워드 추출
    keywords = extract_keywords(query)
    fts_query = " OR ".join(keywords)

    # FTS5 전문 검색 + BM25 랭킹
    cursor.execute("""
        SELECT id, query, answer, bm25(conversations_fts) as score
        FROM conversations_fts
        JOIN conversations ON rowid = id
        WHERE conversations_fts MATCH ?
        ORDER BY score LIMIT ?
    """, (fts_query, limit))
    return cursor.fetchall()`,

    7: `# 하이브리드 검색
async def hybrid_search(query_emb, query_text, top_k=20):
    # Dense search (70%)
    dense_results = await qdrant.search(
        collection="papers", query_vector=query_emb, limit=top_k
    )
    # Sparse search (30%)
    sparse_results = await bm25_search(query_text, limit=top_k)

    # Score fusion
    return fuse_scores(dense_results, sparse_results, weights=[0.7, 0.3])`,

    8: `# GraphDB 검색 (Neo4j)
def graph_enhanced_search(seed_pmids: list, max_depth: int = 2):
    query = """
    UNWIND $seed_pmids AS seed_pmid
    MATCH (seed:Paper {pmid: seed_pmid})
    CALL {
        WITH seed
        MATCH (seed)-[:CITES*1..2]-(related:Paper)  // 인용 관계
        RETURN related, 'citation' AS path_type, 1.0 AS score
        UNION
        MATCH (seed)-[:HAS_KEYWORD]->(k)<-[:HAS_KEYWORD]-(related)  // 키워드 관계
        RETURN related, 'keyword' AS path_type, 0.8 AS score
        UNION
        MATCH (seed)<-[:AUTHORED]-(a)-[:AUTHORED]->(related)  // 저자 관계
        RETURN related, 'author' AS path_type, 0.7 AS score
    }
    RETURN DISTINCT related.pmid, related.title, sum(score) AS relevance
    ORDER BY relevance DESC LIMIT 10
    """
    return neo4j.run(query, seed_pmids=seed_pmids)`,

    9: `# Reranking (Cross-Encoder)
async def rerank_results(query: str, candidates: list, top_k: int = 5):
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # 쿼리-문서 쌍 점수 계산
    pairs = [(query, doc.text) for doc in candidates]
    scores = reranker.predict(pairs)

    # 점수 기준 재정렬
    reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in reranked[:top_k]]`,

    10: `# 컨텍스트 구성 (논문 + 메모리 + 그래프)
def build_context(papers: list, memories: list, graph_context: dict) -> str:
    context = ["📚 관련 논문:"]
    for paper in papers[:5]:
        context.append(f"[PMID:{paper.pmid}] {paper.title}\\n{paper.abstract[:300]}...")

    if graph_context:
        context.append("\\n🕸️ 그래프 연결:")
        context.append(f"  인용 연결: {graph_context.get('citations', 0)}개")
        context.append(f"  관련 저자: {graph_context.get('authors', 0)}명")

    if memories:
        context.append("\\n🧠 관련 과거 대화:")
        for mem in memories:
            context.append(f"Q: {mem.query}\\nA: {mem.answer[:200]}...")

    return "\\n".join(context)`,

    11: `# LLM 응답 생성
async def generate_answer(question: str, context: str) -> str:
    response = await openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\\n{context}\\n\\nQuestion: {question}"}
        ]
    )
    return response.choices[0].message.content`,

    12: `# 메모리 저장 (SQLite)
def save_conversation(query: str, answer: str, sources: list):
    query_hash = hashlib.md5(query.lower().strip().encode()).hexdigest()
    sources_str = ",".join(sources) if sources else ""

    cursor.execute("""
        INSERT INTO conversations (query, answer, query_hash, sources_used)
        VALUES (?, ?, ?, ?)
    """, (query, answer, query_hash, sources_str))
    conn.commit()  # FTS 트리거가 자동으로 인덱스 업데이트
    return cursor.lastrowid`,
  }

  return (
    <div className="rounded-lg overflow-hidden bg-slate-900/80 border border-white/10">
      <div className="flex items-center gap-2 px-4 py-2 bg-white/5 border-b border-white/10">
        <div className="w-3 h-3 rounded-full bg-red-500" />
        <div className="w-3 h-3 rounded-full bg-yellow-500" />
        <div className="w-3 h-3 rounded-full bg-green-500" />
        <span className="ml-2 text-xs text-white/50">{step.name.toLowerCase().replace(/ /g, '_')}.py</span>
      </div>
      <pre className="p-4 overflow-x-auto text-xs leading-relaxed">
        <code className="text-cyan-300/90">{codeExamples[step.id]}</code>
      </pre>
    </div>
  )
}
