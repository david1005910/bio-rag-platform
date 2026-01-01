import { useState, useRef, useEffect, useMemo, useCallback } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Send, Loader2, BookOpen, RefreshCw, Database, Search, ExternalLink, FileText, ChevronUp, ChevronDown, Brain, Volume2, VolumeX, Square } from 'lucide-react'
import { Link } from 'react-router-dom'
import DOMPurify from 'dompurify'
import { chatApi } from '@/services/api'
import { useChatStore } from '@/store/chatStore'
import { validateChatMessage } from '@/utils/validation'
import type { ChatMessage } from '@/types'

interface ExtendedChatMessage extends ChatMessage {
  vectordbUsed?: boolean
  searchMode?: string
  memoryUsed?: boolean
  similarQuestionsFound?: number
}

export default function ChatPage() {
  const [input, setInput] = useState('')
  const [useVectordb, setUseVectordb] = useState(true)
  const [useMemory, setUseMemory] = useState(true)
  const [searchMode, setSearchMode] = useState<'hybrid' | 'dense' | 'sparse'>('hybrid')
  const [voiceEnabled, setVoiceEnabled] = useState(false)
  const [speakingMessageId, setSpeakingMessageId] = useState<string | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const {
    messages,
    currentSessionId,
    isLoading,
    addMessage,
    setLoading,
    clearMessages,
  } = useChatStore()

  // 최종답변 추출 함수 (제목과 인용 논문 제외)
  const extractFinalAnswer = (text: string): string => {
    // "4. 최종 답변" 또는 "## 4." 또는 "Final Answer" 섹션 찾기
    const patterns = [
      /(?:##\s*)?(?:✨\s*)?4\.\s*(?:최종\s*)?답변[^]*?(?=(?:##\s*)?5\.|---|\*\*\*|$)/i,
      /(?:##\s*)?(?:최종|Final)\s*(?:답변|Answer)[^]*?(?=(?:##\s*)?\d+\.|---|\*\*\*|$)/i,
      /4\.[^]*?(?=5\.|---|\*\*\*|$)/i,
    ]

    for (const pattern of patterns) {
      const match = text.match(pattern)
      if (match && match[0].length > 50) {
        let result = match[0]
          // 제목 제거 (4. 최종 답변, ## ✨ 4. 최종 답변 등)
          .replace(/^(?:##\s*)?(?:✨\s*)?4\.\s*(?:최종\s*)?(?:답변|Final\s*Answer)[^\n]*/i, '')
          // 인용된 논문 섹션 제거
          .replace(/(?:###?\s*)?📚\s*인용된?\s*논문[^]*/i, '')
          .replace(/(?:###?\s*)?(?:References?|Citations?|인용\s*문헌)[^]*/i, '')
          // ### 서브 제목 라인 제거 (### 핵심 요약, ### 상세 설명 등)
          .replace(/^###[^\n]*$/gm, '')
          // 괄호 안 내용 제거 (예: (PMID:12345), (참고) 등)
          .replace(/\([^)]*\)/g, '')
          // 마크다운 기호 제거
          .replace(/#{1,6}\s*/g, '')
          .replace(/\*\*/g, '')
          .replace(/📌|📖|⚠️|💡/g, '')  // 이모지 제거
          .replace(/\n+/g, ' ')
          .trim()
        return result
      }
    }

    // 패턴을 찾지 못하면 마지막 단락 반환
    const paragraphs = text.split(/\n\n+/).filter(p => p.trim().length > 50)
    return paragraphs.length > 0 ? paragraphs[paragraphs.length - 1].replace(/\*\*/g, '').trim() : text
  }

  // 언어 감지 함수 (한국어 비율 체크)
  const detectLanguage = (text: string): 'ko' | 'en' => {
    const koreanRegex = /[가-힣]/g
    const koreanMatches = text.match(koreanRegex) || []
    const totalChars = text.replace(/\s/g, '').length
    const koreanRatio = koreanMatches.length / totalChars
    return koreanRatio > 0.3 ? 'ko' : 'en'
  }

  // 음성 합성 함수 (언어에 따라 음성 자동 선택)
  const speak = useCallback((text: string, messageId: string) => {
    if (typeof window === 'undefined' || !window.speechSynthesis) return

    // 이전 음성 중지
    window.speechSynthesis.cancel()

    // 최종답변만 추출
    const finalAnswer = extractFinalAnswer(text)

    // 언어 감지
    const language = detectLanguage(finalAnswer)

    const utterance = new SpeechSynthesisUtterance(finalAnswer)
    utterance.lang = language === 'ko' ? 'ko-KR' : 'en-US'
    utterance.rate = 1.1  // 약간 빠른 속도
    utterance.pitch = language === 'ko' ? 1.1 : 1.5  // 한국어(Yuna): 1.1, 영어(Shelley): 1.5
    utterance.volume = 1.0

    // 언어에 따라 음성 선택: 한국어=Yuna, 영어=Sandy
    const voices = window.speechSynthesis.getVoices()
    let selectedVoice

    if (language === 'ko') {
      // 한국어: Yuna 음성
      selectedVoice = voices.find(voice =>
        voice.name.toLowerCase().includes('yuna')
      ) || voices.find(voice =>
        voice.lang.includes('ko')
      )
    } else {
      // 영어: Shelley 음성
      selectedVoice = voices.find(voice =>
        voice.name.toLowerCase().includes('shelley') && voice.lang.includes('en')
      ) || voices.find(voice =>
        voice.lang.includes('en') && voice.name.toLowerCase().includes('female')
      ) || voices.find(voice =>
        voice.lang.includes('en')
      )
    }

    if (selectedVoice) {
      utterance.voice = selectedVoice
    }

    utterance.onstart = () => setSpeakingMessageId(messageId)
    utterance.onend = () => setSpeakingMessageId(null)
    utterance.onerror = () => setSpeakingMessageId(null)

    window.speechSynthesis.speak(utterance)
  }, [])

  // 음성 중지 함수
  const stopSpeaking = useCallback(() => {
    if (typeof window !== 'undefined' && window.speechSynthesis) {
      window.speechSynthesis.cancel()
      setSpeakingMessageId(null)
    }
  }, [])

  // 컴포넌트 언마운트 시 음성 중지
  useEffect(() => {
    return () => {
      if (typeof window !== 'undefined' && window.speechSynthesis) {
        window.speechSynthesis.cancel()
      }
    }
  }, [])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const queryMutation = useMutation({
    mutationFn: (question: string) =>
      chatApi.query(question, currentSessionId || undefined, undefined, {
        useVectordb,
        searchMode,
        useMemory,
      }),
    onMutate: () => {
      setLoading(true)
    },
    onSuccess: (data) => {
      // Deduplicate sources by PMID, keeping the one with highest relevance score
      const uniqueSources = data.sources
        ? Array.from(
            data.sources.reduce((map, source) => {
              const existing = map.get(source.pmid)
              if (!existing || source.relevance > existing.relevance) {
                map.set(source.pmid, source)
              }
              return map
            }, new Map())
          ).map(([, source]) => source)
        : []

      const messageId = Date.now().toString()
      const assistantMessage: ExtendedChatMessage = {
        id: messageId,
        role: 'assistant',
        content: data.answer,
        sources: uniqueSources,
        createdAt: new Date().toISOString(),
        vectordbUsed: data.vectordbUsed,
        searchMode: data.searchMode,
        memoryUsed: data.memoryUsed,
        similarQuestionsFound: data.similarQuestionsFound,
      }
      addMessage(assistantMessage)
      setLoading(false)

      // 음성 읽기 활성화 시 자동으로 최종 답변 읽기
      if (voiceEnabled) {
        setTimeout(() => speak(data.answer, messageId), 500)
      }
    },
    onError: () => {
      const errorMessage: ChatMessage = {
        id: Date.now().toString(),
        role: 'assistant',
        content: '죄송합니다. 답변을 생성하는 중 오류가 발생했습니다. 다시 시도해주세요.',
        createdAt: new Date().toISOString(),
      }
      addMessage(errorMessage)
      setLoading(false)
    },
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (isLoading) return

    // Validate and sanitize input
    const validation = validateChatMessage(input)
    if (!validation.valid || !validation.sanitized) {
      return
    }

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: validation.sanitized,
      createdAt: new Date().toISOString(),
    }

    addMessage(userMessage)
    queryMutation.mutate(validation.sanitized)
    setInput('')
  }

  // Collect all unique sources from messages
  const allSources = useMemo(() => {
    const sourcesMap = new Map<string, { pmid: string; title: string; relevance: number }>()
    messages.forEach((msg) => {
      if (msg.sources) {
        msg.sources.forEach((source) => {
          const existing = sourcesMap.get(source.pmid)
          if (!existing || source.relevance > existing.relevance) {
            sourcesMap.set(source.pmid, {
              pmid: source.pmid,
              title: source.title,
              relevance: source.relevance,
            })
          }
        })
      }
    })
    return Array.from(sourcesMap.values()).sort((a, b) => b.relevance - a.relevance)
  }, [messages])

  const [showSourcesPanel, setShowSourcesPanel] = useState(true)

  return (
    <div className="flex gap-4 h-[calc(100vh-12rem)] mx-4">
      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <div className="glossy-panel px-6 py-4 mb-4">
        <div className="flex items-center justify-between mb-3">
          <div>
            <h1 className="text-xl font-bold liquid-text">AI 논문 Q&A</h1>
            <p className="text-sm liquid-text-muted">
              바이오메디컬 연구에 관한 질문을 하세요
            </p>
          </div>
          <button
            onClick={clearMessages}
            className="glossy-btn flex items-center gap-2 px-3 py-2 text-sm"
          >
            <RefreshCw size={16} />
            새 대화
          </button>
        </div>

        {/* VectorDB & Memory Controls */}
        <div className="flex flex-wrap items-center gap-4 pt-3 border-t border-white/10">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={useVectordb}
              onChange={(e) => setUseVectordb(e.target.checked)}
              className="w-4 h-4 rounded accent-cyan-500"
            />
            <Database size={16} className={useVectordb ? 'text-purple-500' : 'text-slate-400'} />
            <span className={`text-sm font-medium ${useVectordb ? 'text-purple-600' : 'text-slate-500'}`}>
              VectorDB 검색
            </span>
          </label>

          {useVectordb && (
            <div className="flex items-center gap-2">
              <Search size={14} className="text-slate-400" />
              <select
                value={searchMode}
                onChange={(e) => setSearchMode(e.target.value as 'hybrid' | 'dense' | 'sparse')}
                className="glossy-input px-2 py-1 text-sm rounded"
              >
                <option value="hybrid">Hybrid (Dense + Sparse)</option>
                <option value="dense">Dense (Qdrant)</option>
                <option value="sparse">Sparse (SPLADE)</option>
              </select>
            </div>
          )}

          <div className="w-px h-5 bg-slate-300 hidden sm:block" />

          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={useMemory}
              onChange={(e) => setUseMemory(e.target.checked)}
              className="w-4 h-4 rounded accent-pink-500"
            />
            <Brain size={16} className={useMemory ? 'text-pink-500' : 'text-slate-400'} />
            <span className={`text-sm font-medium ${useMemory ? 'text-pink-600' : 'text-slate-500'}`}>
              대화 기억
            </span>
          </label>

          <div className="w-px h-5 bg-slate-300 hidden sm:block" />

          <button
            onClick={() => {
              if (voiceEnabled) stopSpeaking()
              setVoiceEnabled(!voiceEnabled)
            }}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg transition-all ${
              voiceEnabled
                ? 'bg-green-100 text-green-600 border border-green-300'
                : 'bg-slate-100 text-slate-500 border border-slate-200'
            } ${speakingMessageId ? 'animate-pulse' : ''}`}
          >
            {voiceEnabled ? <Volume2 size={16} /> : <VolumeX size={16} />}
            <span className="text-sm font-medium">음성 읽기</span>
          </button>

                  </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 glossy-panel-sm mb-4">
        {messages.length === 0 && (
          <div className="text-center py-16">
            <BookOpen className="mx-auto text-purple-300 mb-4" size={64} />
            <h3 className="text-xl font-medium liquid-text mb-2">
              무엇이든 물어보세요
            </h3>
            <p className="liquid-text-muted max-w-md mx-auto mb-8">
              예시: "CRISPR-Cas9의 off-target 효과를 줄이는 최신 방법은?"
            </p>
            <div className="flex flex-wrap gap-2 justify-center max-w-2xl mx-auto">
              {[
                'CAR-T 세포치료의 최신 동향은?',
                '암 면역치료의 주요 부작용은?',
                'mRNA 백신의 작동 원리는?',
              ].map((suggestion) => (
                <button
                  key={suggestion}
                  onClick={() => setInput(suggestion)}
                  className="glossy-btn px-4 py-2 text-sm hover:scale-105 transition-all"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((message, index) => {
          const isLastAssistant = message.role === 'assistant' &&
            index === messages.length - 1 ||
            (index === messages.length - 2 && messages[messages.length - 1]?.role === 'user')
          return (
            <MessageBubble
              key={message.id}
              message={message}
              voiceEnabled={voiceEnabled}
              isSpeaking={speakingMessageId === message.id}
              onSpeak={() => speak(message.content, message.id)}
              onStop={stopSpeaking}
              showVoiceButton={isLastAssistant}
            />
          )
        })}

        {isLoading && (
          <div className="flex justify-start">
            <div className="glossy-panel-sm px-6 py-4">
              <div className="flex items-center gap-2 liquid-text-muted">
                <Loader2 className="animate-spin" size={18} />
                답변 생성 중...
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

        {/* Input */}
        <div className="glossy-panel p-4">
          <form onSubmit={handleSubmit} className="flex gap-4">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="질문을 입력하세요..."
              className="glossy-input flex-1 px-4 py-3"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className="glossy-btn-primary px-6 py-3 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Send size={20} />
            </button>
          </form>
        </div>
      </div>

      {/* Right Sidebar - Referenced Papers */}
      <div className="w-72 flex-shrink-0 hidden lg:block">
        <div className="glossy-panel p-4 h-full flex flex-col">
          {/* Header */}
          <button
            onClick={() => setShowSourcesPanel(!showSourcesPanel)}
            className="w-full flex items-center justify-between mb-3 flex-shrink-0"
          >
            <div className="flex items-center gap-2">
              <FileText className="text-purple-500" size={20} />
              <h3 className="font-semibold text-slate-800">참고 논문</h3>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs bg-purple-100 text-purple-600 px-2 py-0.5 rounded-full font-medium">
                {allSources.length}
              </span>
              {showSourcesPanel ? (
                <ChevronUp size={18} className="text-slate-400" />
              ) : (
                <ChevronDown size={18} className="text-slate-400" />
              )}
            </div>
          </button>

          {/* Paper List */}
          {showSourcesPanel && (
            <div className="flex-1 overflow-y-auto space-y-2 min-h-0">
              {allSources.length > 0 ? (
                allSources.map((source, index) => (
                  <Link
                    key={source.pmid}
                    to={`/paper/${source.pmid}`}
                    className="flex items-start gap-2 p-3 bg-white/50 rounded-lg border border-slate-200 hover:bg-purple-50 hover:border-purple-300 transition-all group"
                  >
                    <span className="flex-shrink-0 w-6 h-6 rounded bg-gradient-to-br from-purple-500 to-pink-500 text-white text-xs font-bold flex items-center justify-center">
                      {index + 1}
                    </span>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-slate-700 font-medium line-clamp-2 group-hover:text-purple-700">
                        {source.title}
                      </p>
                      <div className="flex items-center gap-2 mt-1">
                        <span className="text-xs text-slate-500">PMID: {source.pmid}</span>
                        <span className="text-xs text-purple-500 font-medium">
                          {(source.relevance * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                    <ExternalLink size={14} className="text-slate-400 group-hover:text-purple-500 flex-shrink-0 mt-1" />
                  </Link>
                ))
              ) : (
                <div className="text-center py-8 text-slate-500">
                  <BookOpen className="mx-auto mb-2 opacity-30" size={32} />
                  <p className="text-sm">질문을 하시면</p>
                  <p className="text-sm">참고 논문이 표시됩니다</p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

interface MessageBubbleProps {
  message: ExtendedChatMessage
  voiceEnabled: boolean
  isSpeaking: boolean
  onSpeak: () => void
  onStop: () => void
  showVoiceButton?: boolean
}

function MessageBubble({ message, voiceEnabled, isSpeaking, onSpeak, onStop, showVoiceButton }: MessageBubbleProps) {
  const isUser = message.role === 'user'

  // Simple markdown-like rendering for AI responses
  const renderContent = (content: string) => {
    if (isUser) {
      return <p className="whitespace-pre-wrap text-white">{content}</p>
    }

    // Configure DOMPurify to only allow safe tags
    const sanitizeConfig = {
      ALLOWED_TAGS: ['strong', 'em', 'b', 'i'],
      ALLOWED_ATTR: ['class'],
    }

    // Process markdown-like formatting for AI responses
    const lines = content.split('\n')
    return (
      <div className="space-y-3 text-slate-700">
        {lines.map((line, idx) => {
          // Bold text: **text**
          const processedLine = line.replace(
            /\*\*(.+?)\*\*/g,
            '<strong class="text-slate-900 font-semibold">$1</strong>'
          )

          // Sanitize HTML to prevent XSS
          const sanitizedLine = DOMPurify.sanitize(processedLine, sanitizeConfig)

          // Numbered list items
          if (/^\d+\.\s/.test(line)) {
            return (
              <div
                key={idx}
                className="pl-4 border-l-2 border-purple-400"
                dangerouslySetInnerHTML={{ __html: sanitizedLine }}
              />
            )
          }

          // Empty lines
          if (!line.trim()) {
            return <div key={idx} className="h-2" />
          }

          // Regular paragraphs
          return (
            <p
              key={idx}
              className="leading-relaxed"
              dangerouslySetInnerHTML={{ __html: sanitizedLine }}
            />
          )
        })}
      </div>
    )
  }

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-3xl px-6 py-4 ${
          isUser
            ? 'glossy-btn-primary rounded-2xl rounded-tr-none'
            : 'glossy-panel rounded-2xl rounded-tl-none'
        }`}
      >
        {/* Main content */}
        {renderContent(message.content)}

        {/* Voice button for last assistant message only */}
        {!isUser && voiceEnabled && showVoiceButton && (
          <div className="mt-3 flex justify-end">
            <button
              onClick={isSpeaking ? onStop : onSpeak}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                isSpeaking
                  ? 'bg-red-100 text-red-600 border border-red-300 animate-pulse'
                  : 'bg-green-100 text-green-600 border border-green-300 hover:bg-green-200'
              }`}
            >
              {isSpeaking ? (
                <>
                  <Square size={14} />
                  정지
                </>
              ) : (
                <>
                  <Volume2 size={14} />
                  읽어주기
                </>
              )}
            </button>
          </div>
        )}

        {/* VectorDB & Memory indicator */}
        {!isUser && (message.vectordbUsed || message.memoryUsed) && (
          <div className="mt-3 flex flex-wrap items-center gap-3 text-xs">
            {message.vectordbUsed && (
              <div className="flex items-center gap-1.5">
                <Database size={12} className="text-purple-500" />
                <span className="text-purple-600 font-medium">
                  VectorDB ({message.searchMode === 'hybrid' ? 'Hybrid' : message.searchMode === 'dense' ? 'Dense' : 'Sparse'})
                </span>
              </div>
            )}
            {message.memoryUsed && (
              <div className="flex items-center gap-1.5">
                <Brain size={12} className="text-pink-500" />
                <span className="text-pink-600 font-medium">
                  대화 기억 사용 ({message.similarQuestionsFound}개 참조)
                </span>
              </div>
            )}
          </div>
        )}

        {/* Sources section */}
        {message.sources && message.sources.length > 0 && (
          <details className="mt-4 pt-4 border-t border-slate-200">
            <summary className="text-sm font-medium text-purple-600 cursor-pointer hover:text-purple-700 transition-colors">
              📚 참고 문헌 ({message.sources.length}개)
              {message.vectordbUsed && <span className="ml-2 text-xs text-purple-400">[VectorDB]</span>}
            </summary>
            <div className="space-y-2 mt-3">
              {message.sources.map((source, index: number) => (
                <div
                  key={source.pmid}
                  className="text-sm bg-slate-50 p-3 rounded-xl border border-slate-200"
                >
                  <div className="font-medium text-slate-800 flex items-center gap-2">
                    [{index + 1}] PMID: {source.pmid}
                    {source.sourceType === 'vectordb' && (
                      <span className="text-[10px] px-1.5 py-0.5 bg-purple-100 text-purple-600 rounded font-medium">
                        VectorDB
                      </span>
                    )}
                  </div>
                  <div className="text-slate-600 text-xs mt-1">{source.title}</div>
                  <div className="text-xs text-slate-500 mt-1 flex flex-wrap gap-3">
                    <span className="text-purple-600 font-medium">Hybrid: {source.relevance.toFixed(2)} <span className="text-slate-400">(가중 평균)</span></span>
                    {source.denseScore !== undefined && source.denseScore !== null && (
                      <span className="text-blue-600 font-medium">Dense: {source.denseScore.toFixed(2)} <span className="text-slate-400">(의미적 유사도)</span></span>
                    )}
                    {source.sparseScore !== undefined && source.sparseScore !== null && (
                      <span className="text-green-600 font-medium">Sparse: {source.sparseScore.toFixed(2)} <span className="text-slate-400">(키워드 매칭)</span></span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </details>
        )}
      </div>
    </div>
  )
}
