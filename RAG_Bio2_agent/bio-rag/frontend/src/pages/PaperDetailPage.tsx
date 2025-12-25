import { useParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { ArrowLeft, ExternalLink, Bookmark, MessageSquare, Share2 } from 'lucide-react'
import { searchApi } from '@/services/api'

export default function PaperDetailPage() {
  const { pmid } = useParams<{ pmid: string }>()

  const { data: paper, isLoading, error } = useQuery({
    queryKey: ['paper', pmid],
    queryFn: () => searchApi.getPaper(pmid!),
    enabled: !!pmid,
  })

  const { data: similarPapers } = useQuery({
    queryKey: ['similarPapers', pmid],
    queryFn: () => searchApi.getSimilarPapers(pmid!),
    enabled: !!pmid,
  })

  if (isLoading) {
    return (
      <div className="flex justify-center py-24">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600" />
      </div>
    )
  }

  if (error || !paper) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-8">
        <div className="bg-red-50 text-red-600 p-6 rounded-xl text-center">
          논문을 찾을 수 없습니다.
          <Link to="/search" className="block mt-4 text-primary-600 hover:underline">
            검색으로 돌아가기
          </Link>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      {/* Back button */}
      <Link
        to="/search"
        className="inline-flex items-center gap-2 text-gray-600 hover:text-gray-900 mb-6"
      >
        <ArrowLeft size={20} />
        검색으로 돌아가기
      </Link>

      {/* Paper header */}
      <div className="bg-white rounded-xl border border-gray-200 p-8 mb-6">
        <h1 className="text-2xl font-bold text-gray-900 mb-4">{paper.title}</h1>

        <div className="flex flex-wrap gap-4 text-sm text-gray-600 mb-6">
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

        {paper.authors.length > 0 && (
          <div className="text-gray-700 mb-6">
            <span className="font-medium">저자: </span>
            {paper.authors.join(', ')}
          </div>
        )}

        <div className="flex flex-wrap gap-3">
          <button className="flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700">
            <Bookmark size={18} />
            저장
          </button>
          <Link
            to={`/chat?pmid=${pmid}`}
            className="flex items-center gap-2 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200"
          >
            <MessageSquare size={18} />
            AI에게 질문
          </Link>
          <button className="flex items-center gap-2 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200">
            <Share2 size={18} />
            공유
          </button>
          {paper.doi && (
            <a
              href={`https://doi.org/${paper.doi}`}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200"
            >
              <ExternalLink size={18} />
              DOI
            </a>
          )}
        </div>
      </div>

      {/* Abstract */}
      <div className="bg-white rounded-xl border border-gray-200 p-8 mb-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">초록</h2>
        <p className="text-gray-700 leading-relaxed whitespace-pre-wrap">
          {paper.abstract || '초록 정보가 없습니다.'}
        </p>
      </div>

      {/* Keywords */}
      {paper.keywords.length > 0 && (
        <div className="bg-white rounded-xl border border-gray-200 p-8 mb-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">키워드</h2>
          <div className="flex flex-wrap gap-2">
            {paper.keywords.map((keyword) => (
              <span
                key={keyword}
                className="px-3 py-1 bg-primary-50 text-primary-700 rounded-full text-sm"
              >
                {keyword}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Similar Papers */}
      {similarPapers && similarPapers.length > 0 && (
        <div className="bg-white rounded-xl border border-gray-200 p-8">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">유사 논문</h2>
          <div className="space-y-4">
            {similarPapers.map((similar: any) => (
              <Link
                key={similar.pmid}
                to={`/paper/${similar.pmid}`}
                className="block p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <div className="font-medium text-gray-900">{similar.title}</div>
                    <div className="text-sm text-gray-500 mt-1">
                      PMID: {similar.pmid}
                    </div>
                    {similar.common_keywords && similar.common_keywords.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-2">
                        {similar.common_keywords.map((kw: string) => (
                          <span key={kw} className="px-2 py-0.5 bg-primary-100 text-primary-700 rounded text-xs">
                            {kw}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                  {similar.similarity_score && (
                    <div className="px-2 py-1 bg-green-100 text-green-700 rounded text-sm font-medium">
                      {Math.round(similar.similarity_score * 100)}%
                    </div>
                  )}
                </div>
              </Link>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
