import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Search, Filter, ExternalLink, MessageSquare, Bookmark } from 'lucide-react'
import { Link } from 'react-router-dom'
import { searchApi } from '@/services/api'
import type { PaperSearchResult } from '@/types'

export default function SearchPage() {
  const [query, setQuery] = useState('')
  const [searchTerm, setSearchTerm] = useState('')

  const { data, isLoading, error } = useQuery({
    queryKey: ['search', searchTerm],
    queryFn: () => searchApi.search(searchTerm),
    enabled: !!searchTerm,
  })

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (query.trim()) {
      setSearchTerm(query.trim())
    }
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">논문 검색</h1>
        <p className="text-gray-600">
          자연어로 검색하면 AI가 의미를 이해하고 관련 논문을 찾아드립니다.
        </p>
      </div>

      {/* Search Bar */}
      <form onSubmit={handleSearch} className="mb-8">
        <div className="flex gap-4">
          <div className="flex-1 relative">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-400" size={20} />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="예: cancer immunotherapy side effects, CRISPR off-target 효과..."
              className="w-full pl-12 pr-4 py-4 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            />
          </div>
          <button
            type="submit"
            disabled={!query.trim()}
            className="px-8 py-4 bg-primary-600 text-white font-medium rounded-xl hover:bg-primary-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            검색
          </button>
        </div>
      </form>

      {/* Results */}
      {isLoading && (
        <div className="flex justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600" />
        </div>
      )}

      {error && (
        <div className="bg-red-50 text-red-600 p-4 rounded-xl">
          검색 중 오류가 발생했습니다. 다시 시도해주세요.
        </div>
      )}

      {data && (
        <div>
          <div className="flex items-center justify-between mb-6">
            <p className="text-gray-600">
              <span className="font-semibold text-gray-900">{data.total}</span>건의 결과
              <span className="text-gray-400 ml-2">({data.tookMs}ms)</span>
            </p>
            <button className="flex items-center gap-2 px-4 py-2 text-gray-600 hover:bg-gray-100 rounded-lg">
              <Filter size={18} />
              필터
            </button>
          </div>

          <div className="space-y-4">
            {data.results.map((paper: PaperSearchResult) => (
              <PaperCard key={paper.pmid} paper={paper} />
            ))}
          </div>

          {data.results.length === 0 && (
            <div className="text-center py-12 text-gray-500">
              검색 결과가 없습니다. 다른 키워드로 검색해보세요.
            </div>
          )}
        </div>
      )}

      {/* Empty state */}
      {!searchTerm && (
        <div className="text-center py-16">
          <Search className="mx-auto text-gray-300 mb-4" size={64} />
          <h3 className="text-xl font-medium text-gray-600 mb-2">
            검색어를 입력하세요
          </h3>
          <p className="text-gray-500 max-w-md mx-auto">
            예시: "What are the latest CAR-T cell therapy developments?"
            또는 "암 면역치료 부작용"
          </p>
        </div>
      )}
    </div>
  )
}

function PaperCard({ paper }: { paper: PaperSearchResult }) {
  const relevancePercent = Math.round(paper.relevanceScore * 100)

  return (
    <div className="bg-white p-6 rounded-xl border border-gray-200 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1">
          <Link
            to={`/paper/${paper.pmid}`}
            className="text-lg font-semibold text-gray-900 hover:text-primary-600 transition-colors"
          >
            {paper.title}
          </Link>
          <div className="flex items-center gap-3 mt-2 text-sm text-gray-500">
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
        <div className="flex items-center gap-1 px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm font-medium">
          {relevancePercent}%
        </div>
      </div>

      <p className="mt-4 text-gray-600 line-clamp-3">
        {paper.abstract}
      </p>

      {paper.authors.length > 0 && (
        <div className="mt-3 text-sm text-gray-500">
          {paper.authors.slice(0, 3).join(', ')}
          {paper.authors.length > 3 && ` 외 ${paper.authors.length - 3}명`}
        </div>
      )}

      <div className="flex items-center gap-4 mt-4 pt-4 border-t border-gray-100">
        <Link
          to={`/paper/${paper.pmid}`}
          className="flex items-center gap-1 text-sm text-primary-600 hover:text-primary-700"
        >
          <ExternalLink size={16} />
          상세보기
        </Link>
        <Link
          to={`/chat?pmid=${paper.pmid}`}
          className="flex items-center gap-1 text-sm text-primary-600 hover:text-primary-700"
        >
          <MessageSquare size={16} />
          AI 질문
        </Link>
        <button className="flex items-center gap-1 text-sm text-gray-500 hover:text-primary-600">
          <Bookmark size={16} />
          저장
        </button>
      </div>
    </div>
  )
}
