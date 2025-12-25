import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Library, Tag, Trash2, ExternalLink, MessageSquare, Loader2, BookOpen, Search, Database, FolderOpen } from 'lucide-react'
import { Link } from 'react-router-dom'
import { libraryApi, vectordbApi } from '@/services/api'
import { useAuthStore } from '@/store/authStore'

type TabType = 'saved' | 'vectordb'

export default function LibraryPage() {
  const { isAuthenticated } = useAuthStore()
  const [selectedTag, setSelectedTag] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<TabType>('vectordb')
  const queryClient = useQueryClient()

  const { data: papersData, isLoading } = useQuery({
    queryKey: ['savedPapers', selectedTag],
    queryFn: () => libraryApi.getSavedPapers(selectedTag || undefined),
    enabled: isAuthenticated && activeTab === 'saved',
  })

  const { data: vectordbPapers, isLoading: isLoadingVectorDB } = useQuery({
    queryKey: ['vectordbPapers'],
    queryFn: () => vectordbApi.getPapers(),
    enabled: activeTab === 'vectordb',
  })

  const { data: tags } = useQuery({
    queryKey: ['tags'],
    queryFn: () => libraryApi.getTags(),
    enabled: isAuthenticated && activeTab === 'saved',
  })

  const deleteMutation = useMutation({
    mutationFn: (paperId: string) => libraryApi.deleteSavedPaper(paperId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['savedPapers'] })
      queryClient.invalidateQueries({ queryKey: ['tags'] })
    },
  })

  const handleDelete = (paperId: string, title: string) => {
    if (confirm(`"${title.substring(0, 50)}..." 논문을 삭제하시겠습니까?`)) {
      deleteMutation.mutate(paperId)
    }
  }

  // Show login prompt only for saved papers tab when not authenticated
  const showLoginPrompt = !isAuthenticated && activeTab === 'saved'

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl font-bold liquid-text">내 라이브러리</h1>
          <p className="liquid-text-muted mt-1">인덱싱된 논문 및 저장한 논문을 관리하세요</p>
        </div>
        <div className="flex items-center gap-2">
          {activeTab === 'vectordb' ? (
            <>
              <Database className="text-cyan-300" size={24} />
              <span className="text-white font-medium">
                {vectordbPapers?.total || 0}개의 인덱싱 논문
              </span>
            </>
          ) : (
            <>
              <BookOpen className="text-cyan-300" size={24} />
              <span className="text-white font-medium">
                {papersData?.length || 0}개의 저장 논문
              </span>
            </>
          )}
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 mb-6">
        <button
          onClick={() => setActiveTab('vectordb')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
            activeTab === 'vectordb'
              ? 'bg-cyan-500/30 text-cyan-200 border border-cyan-400/30'
              : 'glossy-btn text-white/70 hover:text-white'
          }`}
        >
          <Database size={18} />
          VectorDB 인덱싱 논문
          {vectordbPapers?.total ? (
            <span className="ml-1 px-2 py-0.5 bg-cyan-500/30 rounded-full text-xs">
              {vectordbPapers.total}
            </span>
          ) : null}
        </button>
        <button
          onClick={() => setActiveTab('saved')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
            activeTab === 'saved'
              ? 'bg-cyan-500/30 text-cyan-200 border border-cyan-400/30'
              : 'glossy-btn text-white/70 hover:text-white'
          }`}
        >
          <FolderOpen size={18} />
          저장된 논문
          {papersData?.length ? (
            <span className="ml-1 px-2 py-0.5 bg-cyan-500/30 rounded-full text-xs">
              {papersData.length}
            </span>
          ) : null}
        </button>
      </div>

      {/* VectorDB Papers Tab */}
      {activeTab === 'vectordb' && (
        <div>
          {isLoadingVectorDB ? (
            <div className="flex justify-center py-12">
              <div className="flex items-center gap-3 text-white/70">
                <Loader2 className="animate-spin" size={24} />
                <span>VectorDB 논문을 불러오는 중...</span>
              </div>
            </div>
          ) : vectordbPapers && vectordbPapers.papers.length > 0 ? (
            <div className="space-y-4">
              {vectordbPapers.papers.map((paper) => (
                <div
                  key={paper.id}
                  className="glossy-panel p-6 hover:scale-[1.01] transition-all duration-300"
                >
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <Link
                          to={`/paper/${paper.pmid}`}
                          className="text-lg font-semibold liquid-text hover:text-yellow-200 transition-colors"
                        >
                          {paper.title}
                        </Link>
                        <span className="px-2 py-0.5 bg-cyan-500/20 text-cyan-300 rounded text-xs border border-cyan-400/30">
                          <Database size={10} className="inline mr-1" />
                          VectorDB
                        </span>
                      </div>
                      <div className="flex items-center gap-3 mt-2 text-sm liquid-text-muted">
                        <span>PMID: {paper.pmid}</span>
                        {paper.journal && (
                          <>
                            <span>|</span>
                            <span>{paper.journal}</span>
                          </>
                        )}
                      </div>
                    </div>
                  </div>

                  <p className="mt-4 liquid-text-muted line-clamp-2">
                    {paper.abstract}
                  </p>

                  {paper.keywords && paper.keywords.length > 0 && (
                    <div className="flex flex-wrap gap-2 mt-3">
                      {paper.keywords.slice(0, 5).map((keyword) => (
                        <span
                          key={keyword}
                          className="px-2 py-1 bg-green-500/20 text-green-200 rounded-full text-xs border border-green-400/30"
                        >
                          {keyword}
                        </span>
                      ))}
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
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="glossy-panel p-12 text-center">
              <Database className="mx-auto text-white/30 mb-4" size={48} />
              <h3 className="text-lg font-medium text-white mb-2">
                인덱싱된 논문이 없습니다
              </h3>
              <p className="liquid-text-muted mb-6">
                논문 검색 후 "논문 인덱싱" 버튼을 클릭하여 VectorDB에 저장하세요
              </p>
              <Link
                to="/search"
                className="glossy-btn-primary inline-flex items-center gap-2 px-6 py-3"
              >
                <Search size={18} />
                논문 검색하기
              </Link>
            </div>
          )}
        </div>
      )}

      {/* Saved Papers Tab */}
      {activeTab === 'saved' && (
        showLoginPrompt ? (
          <div className="glossy-panel p-12 text-center">
            <Library className="mx-auto text-white/30 mb-4" size={64} />
            <h2 className="text-2xl font-bold text-white mb-2">로그인이 필요합니다</h2>
            <p className="liquid-text-muted mb-6">
              논문을 저장하고 관리하려면 로그인하세요.
            </p>
            <Link
              to="/login"
              className="glossy-btn-primary inline-flex items-center px-6 py-3"
            >
              로그인
            </Link>
          </div>
        ) : (
        <div className="grid md:grid-cols-4 gap-6">
          {/* Sidebar - Tags */}
          <div className="md:col-span-1">
            <div className="glossy-panel p-4">
              <h3 className="font-medium text-white mb-3 flex items-center gap-2">
                <Tag size={16} className="text-cyan-300" />
                태그
              </h3>
              <div className="space-y-1">
                <button
                  onClick={() => setSelectedTag(null)}
                  className={`flex items-center gap-2 w-full px-3 py-2 text-sm rounded-lg transition-all ${
                    selectedTag === null
                      ? 'bg-cyan-500/30 text-cyan-200 border border-cyan-400/30'
                      : 'text-white/70 hover:bg-white/10'
                  }`}
                >
                  전체 보기
                </button>
                {tags && tags.length > 0 ? (
                  tags.map((tag) => (
                    <button
                      key={tag}
                      onClick={() => setSelectedTag(tag)}
                      className={`flex items-center gap-2 w-full px-3 py-2 text-sm rounded-lg transition-all ${
                        selectedTag === tag
                          ? 'bg-cyan-500/30 text-cyan-200 border border-cyan-400/30'
                          : 'text-white/70 hover:bg-white/10'
                      }`}
                    >
                      <Tag size={14} />
                      {tag}
                    </button>
                  ))
                ) : (
                  <p className="text-sm text-white/50 px-3 py-2">태그가 없습니다</p>
                )}
              </div>
            </div>
          </div>

          {/* Papers List */}
          <div className="md:col-span-3">
            {isLoading ? (
              <div className="flex justify-center py-12">
                <div className="flex items-center gap-3 text-white/70">
                  <Loader2 className="animate-spin" size={24} />
                  <span>논문을 불러오는 중...</span>
                </div>
              </div>
            ) : papersData && papersData.length > 0 ? (
              <div className="space-y-4">
                {papersData.map((paper) => (
                  <div
                    key={paper.id}
                    className="glossy-panel p-6 hover:scale-[1.01] transition-all duration-300"
                  >
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
                          {paper.journal && (
                            <>
                              <span>|</span>
                              <span>{paper.journal}</span>
                            </>
                          )}
                        </div>
                      </div>
                    </div>

                    <p className="mt-4 liquid-text-muted line-clamp-2">
                      {paper.abstract}
                    </p>

                    {paper.tags && paper.tags.length > 0 && (
                      <div className="flex flex-wrap gap-2 mt-3">
                        {paper.tags.map((tag) => (
                          <span
                            key={tag}
                            className="px-2 py-1 bg-cyan-500/20 text-cyan-200 rounded-full text-xs border border-cyan-400/30"
                          >
                            {tag}
                          </span>
                        ))}
                      </div>
                    )}

                    {paper.notes && (
                      <div className="mt-3 p-3 bg-white/5 rounded-lg border border-white/10">
                        <p className="text-sm text-white/70">{paper.notes}</p>
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
                        onClick={() => handleDelete(paper.id, paper.title)}
                        disabled={deleteMutation.isPending}
                        className="flex items-center gap-1 text-sm text-red-400 hover:text-red-300 transition-colors ml-auto"
                      >
                        {deleteMutation.isPending ? (
                          <Loader2 size={16} className="animate-spin" />
                        ) : (
                          <Trash2 size={16} />
                        )}
                        삭제
                      </button>
                    </div>
                  </div>
              ))}
            </div>
          ) : (
            <div className="glossy-panel p-12 text-center">
              <Library className="mx-auto text-white/30 mb-4" size={48} />
              <h3 className="text-lg font-medium text-white mb-2">
                {selectedTag ? `"${selectedTag}" 태그의 논문이 없습니다` : '저장된 논문이 없습니다'}
              </h3>
              <p className="liquid-text-muted mb-6">
                검색에서 논문을 찾아 저장해보세요
              </p>
              <Link
                to="/search"
                className="glossy-btn-primary inline-flex items-center gap-2 px-6 py-3"
              >
                <Search size={18} />
                논문 검색하기
              </Link>
            </div>
          )}
          </div>
        </div>
        )
      )}
    </div>
  )
}
