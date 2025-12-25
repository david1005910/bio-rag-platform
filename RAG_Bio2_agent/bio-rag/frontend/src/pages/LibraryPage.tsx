import { useQuery } from '@tanstack/react-query'
import { Library, Tag, Trash2, ExternalLink } from 'lucide-react'
import { Link } from 'react-router-dom'
import { libraryApi } from '@/services/api'
import { useAuthStore } from '@/store/authStore'

export default function LibraryPage() {
  const { isAuthenticated } = useAuthStore()

  const { data: papers, isLoading } = useQuery({
    queryKey: ['savedPapers'],
    queryFn: () => libraryApi.getSavedPapers(),
    enabled: isAuthenticated,
  })

  const { data: tags } = useQuery({
    queryKey: ['tags'],
    queryFn: () => libraryApi.getTags(),
    enabled: isAuthenticated,
  })

  if (!isAuthenticated) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-16 text-center">
        <Library className="mx-auto text-gray-300 mb-4" size={64} />
        <h2 className="text-2xl font-bold text-gray-900 mb-2">로그인이 필요합니다</h2>
        <p className="text-gray-600 mb-6">
          논문을 저장하고 관리하려면 로그인하세요.
        </p>
        <Link
          to="/login"
          className="inline-flex items-center px-6 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700"
        >
          로그인
        </Link>
      </div>
    )
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">내 라이브러리</h1>
          <p className="text-gray-600 mt-1">저장한 논문을 관리하세요</p>
        </div>
      </div>

      <div className="grid md:grid-cols-4 gap-6">
        {/* Sidebar */}
        <div className="md:col-span-1">
          <div className="bg-white rounded-xl border border-gray-200 p-4">
            <h3 className="font-medium text-gray-900 mb-3">태그</h3>
            {tags && tags.length > 0 ? (
              <div className="space-y-1">
                {tags.map((tag) => (
                  <button
                    key={tag}
                    className="flex items-center gap-2 w-full px-3 py-2 text-sm text-gray-600 hover:bg-gray-100 rounded-lg"
                  >
                    <Tag size={14} />
                    {tag}
                  </button>
                ))}
              </div>
            ) : (
              <p className="text-sm text-gray-500">태그가 없습니다</p>
            )}
          </div>
        </div>

        {/* Papers */}
        <div className="md:col-span-3">
          {isLoading ? (
            <div className="flex justify-center py-12">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600" />
            </div>
          ) : papers && papers.length > 0 ? (
            <div className="space-y-4">
              {papers.map((paper) => (
                <div
                  key={paper.id}
                  className="bg-white rounded-xl border border-gray-200 p-6"
                >
                  <Link
                    to={`/paper/${paper.pmid}`}
                    className="text-lg font-semibold text-gray-900 hover:text-primary-600"
                  >
                    {paper.title}
                  </Link>
                  <div className="text-sm text-gray-500 mt-1">
                    PMID: {paper.pmid}
                  </div>
                  <p className="text-gray-600 mt-3 line-clamp-2">
                    {paper.abstract}
                  </p>

                  {paper.tags.length > 0 && (
                    <div className="flex flex-wrap gap-2 mt-3">
                      {paper.tags.map((tag) => (
                        <span
                          key={tag}
                          className="px-2 py-1 bg-gray-100 text-gray-600 rounded text-xs"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  )}

                  <div className="flex items-center gap-4 mt-4 pt-4 border-t border-gray-100">
                    <Link
                      to={`/paper/${paper.pmid}`}
                      className="flex items-center gap-1 text-sm text-primary-600 hover:text-primary-700"
                    >
                      <ExternalLink size={14} />
                      보기
                    </Link>
                    <button className="flex items-center gap-1 text-sm text-red-500 hover:text-red-600">
                      <Trash2 size={14} />
                      삭제
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-12 bg-white rounded-xl border border-gray-200">
              <Library className="mx-auto text-gray-300 mb-4" size={48} />
              <h3 className="text-lg font-medium text-gray-600 mb-2">
                저장된 논문이 없습니다
              </h3>
              <p className="text-gray-500 mb-4">
                검색에서 논문을 찾아 저장해보세요
              </p>
              <Link
                to="/search"
                className="inline-flex items-center px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700"
              >
                논문 검색하기
              </Link>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
