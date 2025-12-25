import { useState, useEffect, useCallback } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Search, Filter, ExternalLink, MessageSquare, Bookmark, BookmarkCheck, Languages, Loader2, Database, CheckCircle } from 'lucide-react'
import { Link, useSearchParams } from 'react-router-dom'
import { searchApi, libraryApi, vectordbApi } from '@/services/api'
import { useAuthStore } from '@/store/authStore'
import type { PaperSearchResult } from '@/types'

// Simple Korean detection
function containsKorean(text: string): boolean {
  return /[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]/.test(text)
}

export default function SearchPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [query, setQuery] = useState(searchParams.get('q') || '')
  const [translatedQuery, setTranslatedQuery] = useState('')
  const [isKoreanSearch, setIsKoreanSearch] = useState(false)
  const [isTranslating, setIsTranslating] = useState(false)
  const [isSavingToVectorDB, setIsSavingToVectorDB] = useState(false)
  const [vectorDBSaveResult, setVectorDBSaveResult] = useState<{ saved: number; chunks: number } | null>(null)

  // Get search term from URL params
  const searchTerm = searchParams.get('q') || ''

  // Translate Korean query using API
  const translateQuery = useCallback(async (koreanText: string) => {
    setIsTranslating(true)
    try {
      const result = await searchApi.translate(koreanText, 'ko', 'en')
      setTranslatedQuery(result.translated)
      return result.translated
    } catch (error) {
      console.error('Translation failed:', error)
      // Fallback: just use original query
      setTranslatedQuery(koreanText)
      return koreanText
    } finally {
      setIsTranslating(false)
    }
  }, [])

  // Initialize query from URL params on mount
  useEffect(() => {
    const urlQuery = searchParams.get('q')
    if (urlQuery) {
      setQuery(urlQuery)
      if (containsKorean(urlQuery)) {
        setIsKoreanSearch(true)
        translateQuery(urlQuery)
      }
    }
  }, [searchParams, translateQuery])

  const { data, isLoading, error } = useQuery({
    queryKey: ['search', isKoreanSearch ? translatedQuery : searchTerm],
    queryFn: () => searchApi.search(isKoreanSearch ? translatedQuery : searchTerm),
    enabled: !!(isKoreanSearch ? translatedQuery : searchTerm) && !isTranslating,
    staleTime: 5 * 60 * 1000, // Cache for 5 minutes
  })

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault()
    if (query.trim()) {
      const trimmedQuery = query.trim()

      // Reset VectorDB save result for new search
      setVectorDBSaveResult(null)

      // Update URL params (this preserves state on back navigation)
      setSearchParams({ q: trimmedQuery })

      // Check if Korean and translate using API
      if (containsKorean(trimmedQuery)) {
        setIsKoreanSearch(true)
        await translateQuery(trimmedQuery)
      } else {
        setIsKoreanSearch(false)
        setTranslatedQuery('')
      }
    }
  }

  // Save all search results to VectorDB
  const handleSaveToVectorDB = async () => {
    if (!data?.results?.length) return

    setIsSavingToVectorDB(true)
    setVectorDBSaveResult(null)

    try {
      const papers = data.results.map((paper: PaperSearchResult) => ({
        pmid: paper.pmid,
        title: paper.title,
        abstract: paper.abstract,
        authors: paper.authors,
        journal: paper.journal,
        publication_date: paper.publicationDate,
        keywords: paper.keywords || [],
      }))

      const result = await vectordbApi.savePapers(papers)
      setVectorDBSaveResult({
        saved: result.saved_count,
        chunks: result.total_chunks,
      })
    } catch (error) {
      console.error('Failed to save to VectorDB:', error)
      alert('VectorDB 저장에 실패했습니다')
    } finally {
      setIsSavingToVectorDB(false)
    }
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold liquid-text mb-2">논문 검색</h1>
        <p className="liquid-text-muted">
          한글 또는 영어로 검색하면 AI가 의미를 이해하고 관련 논문을 찾아드립니다.
        </p>
      </div>

      {/* Search Bar */}
      <form onSubmit={handleSearch} className="mb-8">
        <div className="flex gap-4">
          <div className="flex-1 relative">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-white/50" size={20} />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="예: 암 면역치료 부작용, CRISPR gene therapy..."
              className="glossy-input w-full pl-12 pr-4 py-4"
            />
          </div>
          <button
            type="submit"
            disabled={!query.trim() || isLoading || isTranslating}
            className="glossy-btn-primary px-8 py-4 font-medium disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {isLoading || isTranslating ? <Loader2 className="animate-spin" size={20} /> : <Search size={20} />}
            {isTranslating ? '번역 중...' : '검색'}
          </button>
        </div>
      </form>

      {/* Translation Notice */}
      {isKoreanSearch && translatedQuery && searchTerm && (
        <div className="glossy-panel-sm bg-cyan-500/20 p-4 mb-6 flex items-center gap-3">
          <Languages className="text-cyan-300" size={20} />
          <div>
            <span className="text-white/70">한글 검색: </span>
            <span className="text-white font-medium">"{searchTerm}"</span>
            <span className="text-white/50 mx-2">→</span>
            <span className="text-white/70">영어 변환: </span>
            <span className="text-cyan-300 font-medium">"{translatedQuery}"</span>
          </div>
        </div>
      )}

      {/* Results */}
      {(isLoading || isTranslating) && (
        <div className="flex justify-center py-12">
          <div className="flex items-center gap-3 text-white/70">
            <Loader2 className="animate-spin" size={24} />
            <span>{isTranslating ? 'AI 번역 중...' : 'PubMed에서 검색 중...'}</span>
          </div>
        </div>
      )}

      {error && (
        <div className="glossy-panel-sm bg-red-500/20 text-white p-4">
          검색 중 오류가 발생했습니다. 다시 시도해주세요.
        </div>
      )}

      {data && (
        <div>
          <div className="flex items-center justify-between mb-6">
            <p className="liquid-text-muted">
              <span className="font-semibold text-white">{data.total.toLocaleString()}</span>건의 결과
              <span className="text-white/50 ml-2">({data.tookMs}ms)</span>
            </p>
            <div className="flex items-center gap-3">
              {/* VectorDB Indexing Button */}
              {data.results.length > 0 && (
                <button
                  onClick={handleSaveToVectorDB}
                  disabled={isSavingToVectorDB || !!vectorDBSaveResult}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all ${
                    vectorDBSaveResult
                      ? 'bg-green-500/30 text-green-200 border border-green-400/30'
                      : 'glossy-btn-primary hover:scale-105'
                  }`}
                  title="논문을 벡터화하여 VectorDB에 저장 (Hybrid Search용)"
                >
                  {isSavingToVectorDB ? (
                    <>
                      <Loader2 size={18} className="animate-spin" />
                      <span>인덱싱 중... (Embedding 생성)</span>
                    </>
                  ) : vectorDBSaveResult ? (
                    <>
                      <CheckCircle size={18} />
                      <span>인덱싱 완료: {vectorDBSaveResult.saved}개 논문 ({vectorDBSaveResult.chunks} chunks)</span>
                    </>
                  ) : (
                    <>
                      <Database size={18} />
                      <span>논문 인덱싱 (VectorDB 저장)</span>
                    </>
                  )}
                </button>
              )}
              <button className="glossy-btn flex items-center gap-2 px-4 py-2">
                <Filter size={18} />
                필터
              </button>
            </div>
          </div>

          <div className="space-y-4">
            {data.results.map((paper: PaperSearchResult) => (
              <PaperCard key={paper.pmid} paper={paper} />
            ))}
          </div>

          {data.results.length === 0 && (
            <div className="text-center py-12 liquid-text-muted">
              검색 결과가 없습니다. 다른 키워드로 검색해보세요.
            </div>
          )}
        </div>
      )}

      {/* Empty state */}
      {!searchTerm && (
        <div className="text-center py-16">
          <Search className="mx-auto text-white/30 mb-4" size={64} />
          <h3 className="text-xl font-medium liquid-text mb-2">
            검색어를 입력하세요
          </h3>
          <p className="liquid-text-muted max-w-md mx-auto mb-6">
            한글 또는 영어로 검색할 수 있습니다.
          </p>
          <div className="flex flex-wrap gap-2 justify-center max-w-2xl mx-auto">
            {[
              '암 면역치료 최신 동향',
              'CRISPR gene therapy',
              '유전자 편집 부작용',
              'CAR-T cell therapy',
            ].map((suggestion) => (
              <button
                key={suggestion}
                onClick={async () => {
                  setQuery(suggestion)
                  setSearchParams({ q: suggestion })
                  if (containsKorean(suggestion)) {
                    setIsKoreanSearch(true)
                    await translateQuery(suggestion)
                  } else {
                    setIsKoreanSearch(false)
                    setTranslatedQuery('')
                  }
                }}
                className="glossy-btn px-4 py-2 text-sm hover:scale-105 transition-all"
              >
                {suggestion}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

interface PaperCardProps {
  paper: PaperSearchResult
}

function PaperCard({ paper }: PaperCardProps) {
  const relevancePercent = Math.round(paper.relevanceScore * 100)
  const [koreanSummary, setKoreanSummary] = useState<string | null>(null)
  const [isLoadingSummary, setIsLoadingSummary] = useState(false)
  const [summaryError, setSummaryError] = useState(false)
  const [showKorean, setShowKorean] = useState(false)
  const [isSaved, setIsSaved] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const { isAuthenticated } = useAuthStore()
  const queryClient = useQueryClient()

  // Handle save paper
  const handleSavePaper = async () => {
    if (!isAuthenticated) {
      alert('로그인이 필요합니다')
      return
    }

    if (isSaved) {
      // Already saved - could implement unsave here
      return
    }

    setIsSaving(true)
    try {
      await libraryApi.savePaper(paper.pmid)
      setIsSaved(true)
      queryClient.invalidateQueries({ queryKey: ['savedPapers'] })
    } catch (error) {
      console.error('Failed to save paper:', error)
      alert('저장에 실패했습니다')
    } finally {
      setIsSaving(false)
    }
  }

  // Fetch Korean summary from API when button is clicked
  const handleKoreanSummaryClick = async () => {
    // If already have summary, just toggle display
    if (koreanSummary) {
      setShowKorean(!showKorean)
      return
    }

    // Fetch Korean summary
    setIsLoadingSummary(true)
    setSummaryError(false)
    setShowKorean(true)

    try {
      const response = await searchApi.summarize(paper.abstract, 'ko')
      setKoreanSummary(response.summary)
    } catch (error) {
      console.error('Failed to fetch Korean summary:', error)
      setSummaryError(true)
      setKoreanSummary(`[요약 실패] ${paper.abstract.substring(0, 200)}...`)
    } finally {
      setIsLoadingSummary(false)
    }
  }

  return (
    <div className="glossy-panel p-6 hover:scale-[1.01] transition-all duration-300">
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1">
          <Link
            to={`/paper/${paper.pmid}`}
            className="text-lg font-semibold liquid-text hover:text-yellow-200 transition-colors"
          >
            {paper.title}
          </Link>
          <div className="flex items-center gap-3 mt-2 text-sm liquid-text-muted">
            <span>PMID: {paper.pmid}</span>
            <span>|</span>
            <span>{paper.journal}</span>
            {paper.publicationDate && (
              <>
                <span>|</span>
                <span>{paper.publicationDate}</span>
              </>
            )}
          </div>
        </div>
        <div className="flex items-center gap-1 px-3 py-1 bg-green-500/30 text-green-200 rounded-full text-sm font-medium border border-green-400/30">
          {relevancePercent}%
        </div>
      </div>

      {/* Abstract with Korean Summary Toggle */}
      <div className="mt-4">
        {/* Toggle Button */}
        <div className="flex items-center gap-2 mb-2">
          <button
            onClick={handleKoreanSummaryClick}
            disabled={isLoadingSummary}
            className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-medium transition-all ${
              showKorean
                ? 'bg-cyan-500/30 text-cyan-200 border border-cyan-400/30'
                : 'bg-white/10 text-white/70 border border-white/20 hover:bg-white/20'
            }`}
          >
            <Languages size={14} />
            {isLoadingSummary ? (
              <>
                <Loader2 size={14} className="animate-spin" />
                <span>번역 중...</span>
              </>
            ) : showKorean ? (
              <span>영문 보기</span>
            ) : (
              <span>한글 요약</span>
            )}
          </button>
          {showKorean && koreanSummary && !isLoadingSummary && (
            <span className="text-xs text-cyan-300/70">AI 번역</span>
          )}
        </div>

        {/* Content */}
        {isLoadingSummary ? (
          <div className="bg-white/5 p-3 rounded-lg border border-white/10 animate-pulse">
            <div className="h-4 bg-white/10 rounded w-full mb-2"></div>
            <div className="h-4 bg-white/10 rounded w-5/6 mb-2"></div>
            <div className="h-4 bg-white/10 rounded w-3/4"></div>
          </div>
        ) : showKorean && koreanSummary ? (
          <p className={`liquid-text-muted line-clamp-4 bg-cyan-500/10 p-3 rounded-lg border ${summaryError ? 'border-red-400/30' : 'border-cyan-400/20'}`}>
            {koreanSummary}
          </p>
        ) : (
          <p className="liquid-text-muted line-clamp-3">
            {paper.abstract}
          </p>
        )}
      </div>

      {paper.authors.length > 0 && (
        <div className="mt-3 text-sm text-white/60">
          {paper.authors.slice(0, 3).join(', ')}
          {paper.authors.length > 3 && ` 외 ${paper.authors.length - 3}명`}
        </div>
      )}

      <div className="flex items-center gap-4 mt-4 pt-4 border-t border-white/10">
        <Link
          to={`/paper/${paper.pmid}`}
          className="flex items-center gap-1 text-sm text-cyan-300 hover:text-cyan-200 transition-colors"
        >
          <ExternalLink size={16} />
          상세보기
        </Link>
        <Link
          to={`/chat?pmid=${paper.pmid}`}
          className="flex items-center gap-1 text-sm text-pink-300 hover:text-pink-200 transition-colors"
        >
          <MessageSquare size={16} />
          AI 질문
        </Link>
        <button
          onClick={handleSavePaper}
          disabled={isSaving || isSaved}
          className={`flex items-center gap-1 text-sm transition-colors ${
            isSaved
              ? 'text-yellow-300'
              : 'text-white/60 hover:text-yellow-300'
          }`}
        >
          {isSaving ? (
            <Loader2 size={16} className="animate-spin" />
          ) : isSaved ? (
            <BookmarkCheck size={16} />
          ) : (
            <Bookmark size={16} />
          )}
          {isSaved ? '저장됨' : '저장'}
        </button>
      </div>
    </div>
  )
}
