import { useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { ArrowLeft, ExternalLink, Bookmark, MessageSquare, Share2, Loader2, FileDown, FileX } from 'lucide-react'
import { searchApi } from '@/services/api'
import type { PDFInfo } from '@/types'

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

  // PDF info state
  const [pdfInfo, setPdfInfo] = useState<PDFInfo | null>(null)
  const [isCheckingPdf, setIsCheckingPdf] = useState(false)

  // Check PDF availability and open
  const handlePdfClick = async () => {
    if (!pmid) return

    if (!pdfInfo) {
      setIsCheckingPdf(true)
      try {
        const info = await searchApi.getPdfInfo(pmid)
        setPdfInfo(info)

        if (info.hasPdf && info.pdfUrl) {
          window.open(info.pdfUrl, '_blank')
        }
      } catch (error) {
        console.error('Failed to check PDF availability:', error)
        setPdfInfo({ pmid, hasPdf: false, isOpenAccess: false })
      } finally {
        setIsCheckingPdf(false)
      }
    } else if (pdfInfo.hasPdf && pdfInfo.pdfUrl) {
      window.open(pdfInfo.pdfUrl, '_blank')
    }
  }

  if (isLoading) {
    return (
      <div className="flex justify-center items-center py-24">
        <div className="flex items-center gap-3 text-white/70">
          <Loader2 className="animate-spin" size={24} />
          <span>논문 정보를 불러오는 중...</span>
        </div>
      </div>
    )
  }

  if (error || !paper) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-8">
        <div className="glossy-panel bg-red-500/20 p-6 text-center">
          <p className="text-white mb-4">논문을 찾을 수 없습니다.</p>
          <Link to="/search" className="text-cyan-300 hover:text-cyan-200 transition-colors">
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
        className="inline-flex items-center gap-2 text-white/70 hover:text-white mb-6 transition-colors"
      >
        <ArrowLeft size={20} />
        검색으로 돌아가기
      </Link>

      {/* Paper header */}
      <div className="glossy-panel p-8 mb-6">
        <h1 className="text-2xl font-bold text-white mb-4">{paper.title}</h1>

        <div className="flex flex-wrap gap-4 text-sm text-white/60 mb-6">
          <span className="px-3 py-1 bg-white/10 rounded-full">PMID: {paper.pmid}</span>
          <span className="px-3 py-1 bg-white/10 rounded-full">{paper.journal}</span>
          {paper.publicationDate && (
            <span className="px-3 py-1 bg-white/10 rounded-full">{paper.publicationDate}</span>
          )}
        </div>

        {paper.authors.length > 0 && (
          <div className="text-white/80 mb-6">
            <span className="font-medium text-cyan-300">저자: </span>
            {paper.authors.join(', ')}
          </div>
        )}

        <div className="flex flex-wrap gap-3">
          <button className="glossy-btn-primary flex items-center gap-2 px-4 py-2">
            <Bookmark size={18} />
            저장
          </button>
          <Link
            to={`/chat?pmid=${pmid}`}
            className="glossy-btn flex items-center gap-2 px-4 py-2"
          >
            <MessageSquare size={18} />
            AI에게 질문
          </Link>
          <button
            onClick={handlePdfClick}
            disabled={isCheckingPdf || (pdfInfo !== null && !pdfInfo.hasPdf)}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl font-medium transition-all ${
              pdfInfo?.hasPdf
                ? 'bg-green-500/30 text-green-200 border border-green-400/30 hover:bg-green-500/40'
                : pdfInfo !== null
                  ? 'bg-white/10 text-white/40 border border-white/10 cursor-not-allowed'
                  : 'glossy-btn hover:scale-105'
            }`}
            title={
              pdfInfo?.hasPdf
                ? `PDF 다운로드 (${pdfInfo.pmcid})`
                : pdfInfo !== null
                  ? 'PMC에서 PDF를 제공하지 않습니다'
                  : 'PDF 이용 가능 여부 확인'
            }
          >
            {isCheckingPdf ? (
              <Loader2 size={18} className="animate-spin" />
            ) : pdfInfo?.hasPdf ? (
              <FileDown size={18} />
            ) : pdfInfo !== null ? (
              <FileX size={18} />
            ) : (
              <FileDown size={18} />
            )}
            {isCheckingPdf
              ? 'PDF 확인 중...'
              : pdfInfo?.hasPdf
                ? 'PDF 보기'
                : pdfInfo !== null
                  ? 'PDF 미제공'
                  : 'PDF'}
          </button>
          <button className="glossy-btn flex items-center gap-2 px-4 py-2">
            <Share2 size={18} />
            공유
          </button>
          {paper.doi && (
            <a
              href={`https://doi.org/${paper.doi}`}
              target="_blank"
              rel="noopener noreferrer"
              className="glossy-btn flex items-center gap-2 px-4 py-2"
            >
              <ExternalLink size={18} />
              DOI
            </a>
          )}
          <a
            href={`https://pubmed.ncbi.nlm.nih.gov/${paper.pmid}/`}
            target="_blank"
            rel="noopener noreferrer"
            className="glossy-btn flex items-center gap-2 px-4 py-2"
          >
            <ExternalLink size={18} />
            PubMed
          </a>
        </div>
      </div>

      {/* Abstract */}
      <div className="glossy-panel p-8 mb-6">
        <h2 className="text-lg font-semibold text-white mb-4">초록 (Abstract)</h2>
        <p className="text-white/80 leading-relaxed whitespace-pre-wrap">
          {paper.abstract || '초록 정보가 없습니다.'}
        </p>
      </div>

      {/* Keywords */}
      {paper.keywords && paper.keywords.length > 0 && (
        <div className="glossy-panel p-8 mb-6">
          <h2 className="text-lg font-semibold text-white mb-4">키워드</h2>
          <div className="flex flex-wrap gap-2">
            {paper.keywords.map((keyword) => (
              <span
                key={keyword}
                className="px-3 py-1 bg-cyan-500/20 text-cyan-200 rounded-full text-sm border border-cyan-400/30"
              >
                {keyword}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* MeSH Terms */}
      {paper.meshTerms && paper.meshTerms.length > 0 && (
        <div className="glossy-panel p-8 mb-6">
          <h2 className="text-lg font-semibold text-white mb-4">MeSH 용어</h2>
          <div className="flex flex-wrap gap-2">
            {paper.meshTerms.map((term) => (
              <span
                key={term}
                className="px-3 py-1 bg-pink-500/20 text-pink-200 rounded-full text-sm border border-pink-400/30"
              >
                {term}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Similar Papers */}
      {similarPapers && similarPapers.length > 0 && (
        <div className="glossy-panel p-8">
          <h2 className="text-lg font-semibold text-white mb-4">유사 논문</h2>
          <div className="space-y-4">
            {similarPapers.map((similar: any) => (
              <Link
                key={similar.pmid}
                to={`/paper/${similar.pmid}`}
                className="block p-4 bg-white/10 rounded-xl hover:bg-white/20 transition-colors border border-white/10"
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <div className="font-medium text-white">{similar.title}</div>
                    <div className="text-sm text-white/50 mt-1">
                      PMID: {similar.pmid}
                    </div>
                    {similar.common_keywords && similar.common_keywords.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-2">
                        {similar.common_keywords.map((kw: string) => (
                          <span key={kw} className="px-2 py-0.5 bg-cyan-500/20 text-cyan-200 rounded text-xs">
                            {kw}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                  {similar.similarity_score && (
                    <div className="px-2 py-1 bg-green-500/30 text-green-200 rounded text-sm font-medium border border-green-400/30">
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
