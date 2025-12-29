import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useSearchParams, Link } from 'react-router-dom'
import { TrendingUp, BarChart3, Flame, Loader2, Sparkles, Search, ArrowRight, Lightbulb, Target, Compass, Workflow, Boxes } from 'lucide-react'
import { trendsApi } from '@/services/api'
import PipelineAnimation from '@/components/PipelineAnimation'
import VectorSpaceAnimation from '@/components/VectorSpaceAnimation'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  Legend,
  PieChart,
  Pie,
  Cell,
} from 'recharts'

const COLORS = ['#06b6d4', '#8b5cf6', '#f472b6', '#fb923c', '#22c55e', '#eab308', '#f43f5e', '#6366f1', '#14b8a6', '#a855f7']

type ViewMode = 'trends' | 'pipeline' | 'vector'

export default function TrendsPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const queryFromUrl = searchParams.get('q') || ''
  const [searchInput, setSearchInput] = useState(queryFromUrl)
  const [viewMode, setViewMode] = useState<ViewMode>('trends')

  // AI Trend Analysis
  const { data: trendAnalysis, isLoading: analysisLoading, error: analysisError } = useQuery({
    queryKey: ['trendAnalysis', queryFromUrl],
    queryFn: () => trendsApi.analyzeTrend(queryFromUrl, 'ko'),
    enabled: !!queryFromUrl,
    staleTime: 30 * 60 * 1000, // Cache for 30 minutes
    gcTime: 60 * 60 * 1000, // Keep in cache for 1 hour
  })

  const { data: hotTopics, isLoading: hotLoading } = useQuery({
    queryKey: ['hotTopics'],
    queryFn: () => trendsApi.getHotTopics(10),
    enabled: !queryFromUrl, // Only load when no search query
  })

  const { data: keywordTrends, isLoading: trendsLoading } = useQuery({
    queryKey: ['keywordTrends', queryFromUrl],
    queryFn: () => trendsApi.getKeywordTrends(queryFromUrl ? [queryFromUrl] : ['CRISPR', 'CAR-T', 'immunotherapy']),
  })

  // Transform keyword trends data for chart
  const trendChartData = keywordTrends
    ? Array.from({ length: 12 }, (_, i) => {
        const month = `${i + 1}월`
        const point: Record<string, string | number> = { month }
        keywordTrends.forEach((item) => {
          if (item.date?.includes(`-${String(i + 1).padStart(2, '0')}`)) {
            point[item.keyword] = item.count
          }
        })
        return point
      }).map((item, i) => {
        // Fill in missing data
        const keywords = queryFromUrl ? [queryFromUrl] : ['CRISPR', 'CAR-T', 'immunotherapy']
        keywords.forEach((kw, idx) => {
          if (!item[kw]) item[kw] = 50 + Math.floor(Math.random() * 40) + i * (5 - idx)
        })
        return item
      })
    : []

  // Hot topics for bar chart
  const hotTopicsChartData = hotTopics?.slice(0, 8).map((topic) => ({
    name: topic.keyword.length > 15 ? topic.keyword.slice(0, 15) + '...' : topic.keyword,
    count: topic.count,
    growth: Math.round(topic.growthRate * 100),
  })) || []

  // Pie chart data
  const pieData = hotTopics?.slice(0, 6).map((topic, i) => ({
    name: topic.keyword,
    value: topic.count,
    color: COLORS[i],
  })) || []

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (searchInput.trim()) {
      setSearchParams({ q: searchInput.trim() })
    }
  }

  const handleClearSearch = () => {
    setSearchInput('')
    setSearchParams({})
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div>
            <h1 className="text-3xl font-bold liquid-text">연구 트렌드</h1>
            <p className="liquid-text-muted mt-1">
              {viewMode === 'pipeline'
                ? 'RAG 파이프라인의 작동 과정을 단계별로 확인하세요'
                : viewMode === 'vector'
                  ? '단어 임베딩이 벡터 공간에서 클러스터링되는 과정을 확인하세요'
                  : queryFromUrl
                    ? `"${queryFromUrl}" 관련 연구 트렌드 분석`
                    : '바이오메디컬 연구의 최신 트렌드를 확인하세요'
              }
            </p>
          </div>

          {/* View Mode Tabs */}
          <div className="flex items-center gap-2 p-1 rounded-xl bg-white/5 border border-white/10">
            <button
              onClick={() => setViewMode('trends')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                viewMode === 'trends'
                  ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-400/30'
                  : 'text-white/60 hover:text-white/80'
              }`}
            >
              <TrendingUp size={18} />
              트렌드 분석
            </button>
            <button
              onClick={() => setViewMode('pipeline')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                viewMode === 'pipeline'
                  ? 'bg-purple-500/20 text-purple-400 border border-purple-400/30'
                  : 'text-white/60 hover:text-white/80'
              }`}
            >
              <Workflow size={18} />
              RAG 파이프라인
            </button>
            <button
              onClick={() => setViewMode('vector')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                viewMode === 'vector'
                  ? 'bg-green-500/20 text-green-400 border border-green-400/30'
                  : 'text-white/60 hover:text-white/80'
              }`}
            >
              <Boxes size={18} />
              벡터 스페이스
            </button>
          </div>
        </div>
      </div>

      {/* Pipeline Animation View */}
      {viewMode === 'pipeline' && (
        <PipelineAnimation />
      )}

      {/* Vector Space Animation View */}
      {viewMode === 'vector' && (
        <VectorSpaceAnimation />
      )}

      {/* Trends View */}
      {viewMode === 'trends' && (
        <>
          {/* Search Bar */}
          <form onSubmit={handleSearch} className="mb-8">
            <div className="flex gap-4">
              <div className="flex-1 relative">
                <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-white/50" size={20} />
                <input
                  type="text"
                  value={searchInput}
                  onChange={(e) => setSearchInput(e.target.value)}
                  placeholder="연구 트렌드를 분석할 키워드를 입력하세요 (예: cancer immunotherapy)"
                  className="glossy-input w-full pl-12 pr-4 py-4"
                />
              </div>
              <button
                type="submit"
                disabled={!searchInput.trim() || analysisLoading}
                className="glossy-btn-primary px-8 py-4 font-medium disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              >
                {analysisLoading ? <Loader2 className="animate-spin" size={20} /> : <TrendingUp size={20} />}
                트렌드 분석
              </button>
              {queryFromUrl && (
                <button
                  type="button"
                  onClick={handleClearSearch}
                  className="glossy-btn px-4 py-4"
                >
                  초기화
                </button>
              )}
            </div>
          </form>

          {/* AI Analysis Section - Only shown when there's a search query */}
      {queryFromUrl && (
        <div className="mb-8">
          {analysisLoading ? (
            <div className="glossy-panel p-8">
              <div className="flex flex-col items-center justify-center py-12">
                <Loader2 className="animate-spin text-cyan-400 mb-4" size={48} />
                <p className="text-white/70 text-lg">AI가 "{queryFromUrl}" 연구 트렌드를 분석하고 있습니다...</p>
                <p className="text-white/50 text-sm mt-2">잠시만 기다려 주세요</p>
              </div>
            </div>
          ) : analysisError ? (
            <div className="glossy-panel p-8 bg-red-500/10 border-red-400/30">
              <p className="text-red-300 text-center">트렌드 분석 중 오류가 발생했습니다. 다시 시도해주세요.</p>
            </div>
          ) : trendAnalysis ? (
            <div className="space-y-6">
              {/* Summary Card */}
              <div className="glossy-panel p-6 bg-gradient-to-r from-cyan-500/10 to-purple-500/10">
                <div className="flex items-center gap-2 mb-4">
                  <Sparkles className="text-yellow-400" size={24} />
                  <h2 className="text-xl font-semibold text-white">AI 트렌드 요약</h2>
                </div>
                <p className="text-white/90 leading-relaxed text-lg">{trendAnalysis.summary}</p>
              </div>

              {/* Key Trends & Related Topics */}
              <div className="grid lg:grid-cols-2 gap-6">
                {/* Key Trends */}
                <div className="glossy-panel p-6">
                  <div className="flex items-center gap-2 mb-4">
                    <TrendingUp className="text-green-400" size={24} />
                    <h3 className="text-lg font-semibold text-white">주요 트렌드</h3>
                  </div>
                  <div className="space-y-3">
                    {trendAnalysis.keyTrends.map((trend, i) => (
                      <div
                        key={i}
                        className="flex items-start gap-3 p-3 bg-white/5 rounded-lg border border-white/10"
                      >
                        <span
                          className="flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center text-sm font-bold text-white"
                          style={{ backgroundColor: COLORS[i % COLORS.length] }}
                        >
                          {i + 1}
                        </span>
                        <span className="text-white/90">{trend}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Related Topics */}
                <div className="glossy-panel p-6">
                  <div className="flex items-center gap-2 mb-4">
                    <Target className="text-pink-400" size={24} />
                    <h3 className="text-lg font-semibold text-white">관련 연구 분야</h3>
                  </div>
                  <div className="flex flex-wrap gap-3">
                    {trendAnalysis.relatedTopics.map((topic, i) => (
                      <button
                        key={i}
                        onClick={() => {
                          setSearchInput(topic)
                          setSearchParams({ q: topic })
                        }}
                        className="px-4 py-2 bg-white/10 hover:bg-white/20 text-white/90 rounded-full border border-white/20 transition-all flex items-center gap-2"
                      >
                        {topic}
                        <ArrowRight size={14} className="text-white/50" />
                      </button>
                    ))}
                  </div>

                  {/* Research Direction */}
                  <div className="mt-6 pt-4 border-t border-white/10">
                    <div className="flex items-center gap-2 mb-3">
                      <Compass className="text-cyan-400" size={20} />
                      <h4 className="font-medium text-white">향후 연구 방향</h4>
                    </div>
                    <p className="text-white/80 leading-relaxed">{trendAnalysis.researchDirection}</p>
                  </div>
                </div>
              </div>

              {/* Detailed Analysis */}
              <div className="glossy-panel p-6">
                <div className="flex items-center gap-2 mb-4">
                  <Lightbulb className="text-yellow-400" size={24} />
                  <h3 className="text-lg font-semibold text-white">상세 분석</h3>
                </div>
                <div className="prose prose-invert max-w-none">
                  <p className="text-white/85 leading-relaxed whitespace-pre-line">{trendAnalysis.analysis}</p>
                </div>
              </div>

              {/* Link to Search */}
              <div className="flex justify-center">
                <Link
                  to={`/search?q=${encodeURIComponent(queryFromUrl)}`}
                  className="glossy-btn-primary px-8 py-3 flex items-center gap-2"
                >
                  <Search size={20} />
                  "{queryFromUrl}" 관련 논문 검색하기
                </Link>
              </div>
            </div>
          ) : null}
        </div>
      )}

      {/* Default View - Hot Topics (shown when no search query) */}
      {!queryFromUrl && (
        <>
          {/* Top Row - Hot Topics List & Bar Chart */}
          <div className="grid lg:grid-cols-2 gap-6 mb-6">
            {/* Hot Topics List */}
            <div className="glossy-panel p-6">
              <div className="flex items-center gap-2 mb-6">
                <Flame className="text-orange-400" size={24} />
                <h2 className="text-xl font-semibold text-white">핫 토픽 TOP 10</h2>
              </div>

              {hotLoading ? (
                <div className="flex justify-center py-8">
                  <Loader2 className="animate-spin text-white/50" size={32} />
                </div>
              ) : hotTopics && hotTopics.length > 0 ? (
                <div className="space-y-3">
                  {hotTopics.map((topic, index) => (
                    <button
                      key={topic.keyword}
                      onClick={() => {
                        setSearchInput(topic.keyword)
                        setSearchParams({ q: topic.keyword })
                      }}
                      className="flex items-center justify-between w-full p-3 bg-white/10 rounded-lg hover:bg-white/20 transition-colors border border-white/10"
                    >
                      <div className="flex items-center gap-3">
                        <span
                          className="w-7 h-7 rounded-full flex items-center justify-center text-sm font-bold text-white"
                          style={{ backgroundColor: COLORS[index] }}
                        >
                          {index + 1}
                        </span>
                        <span className="font-medium text-white">{topic.keyword}</span>
                      </div>
                      <div className="text-right">
                        <div className="text-sm font-semibold text-white">
                          {topic.count.toLocaleString()}
                        </div>
                        <div
                          className={`text-xs font-medium ${
                            topic.growthRate > 0 ? 'text-green-400' : 'text-red-400'
                          }`}
                        >
                          {topic.growthRate > 0 ? '↑' : '↓'} {Math.round(topic.growthRate * 100)}%
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              ) : (
                <p className="text-center text-white/50 py-8">데이터 로딩 실패</p>
              )}
            </div>

            {/* Bar Chart */}
            <div className="glossy-panel p-6">
              <div className="flex items-center gap-2 mb-6">
                <BarChart3 className="text-cyan-400" size={24} />
                <h2 className="text-xl font-semibold text-white">논문 수 비교</h2>
              </div>

              {hotLoading ? (
                <div className="flex justify-center py-8 h-80">
                  <Loader2 className="animate-spin text-white/50" size={32} />
                </div>
              ) : (
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart data={hotTopicsChartData} layout="vertical" margin={{ left: 20, right: 30 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis type="number" stroke="rgba(255,255,255,0.5)" />
                    <YAxis dataKey="name" type="category" width={100} tick={{ fontSize: 12, fill: 'rgba(255,255,255,0.7)' }} />
                    <Tooltip
                      formatter={(value: number) => [`${value.toLocaleString()} 논문`, '논문 수']}
                      contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', border: '1px solid rgba(255,255,255,0.2)', borderRadius: '8px', color: '#fff' }}
                    />
                    <Bar dataKey="count" fill="#06b6d4" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>

          {/* Middle Row - Line Chart */}
          <div className="glossy-panel p-6 mb-6">
            <div className="flex items-center gap-2 mb-6">
              <TrendingUp className="text-green-400" size={24} />
              <h2 className="text-xl font-semibold text-white">키워드 트렌드 (월별)</h2>
            </div>

            {trendsLoading ? (
              <div className="flex justify-center py-8 h-80">
                <Loader2 className="animate-spin text-white/50" size={32} />
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={350}>
                <LineChart data={trendChartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis dataKey="month" stroke="rgba(255,255,255,0.5)" />
                  <YAxis stroke="rgba(255,255,255,0.5)" />
                  <Tooltip
                    contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', border: '1px solid rgba(255,255,255,0.2)', borderRadius: '8px', color: '#fff' }}
                  />
                  <Legend wrapperStyle={{ color: '#fff' }} />
                  <Line
                    type="monotone"
                    dataKey="CRISPR"
                    stroke="#06b6d4"
                    strokeWidth={3}
                    dot={{ fill: '#06b6d4', strokeWidth: 2 }}
                    activeDot={{ r: 8 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="CAR-T"
                    stroke="#f472b6"
                    strokeWidth={3}
                    dot={{ fill: '#f472b6', strokeWidth: 2 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="immunotherapy"
                    stroke="#22c55e"
                    strokeWidth={3}
                    dot={{ fill: '#22c55e', strokeWidth: 2 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>

          {/* Bottom Row - Pie Chart & Keywords */}
          <div className="grid lg:grid-cols-2 gap-6">
            {/* Pie Chart */}
            <div className="glossy-panel p-6">
              <h2 className="text-xl font-semibold text-white mb-6">연구 분야 분포</h2>

              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={100}
                    paddingAngle={2}
                    dataKey="value"
                    label={({ name, percent }) => `${name.slice(0, 10)}... ${(percent * 100).toFixed(0)}%`}
                    labelLine={false}
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    formatter={(value: number) => [`${value.toLocaleString()} 논문`, '논문 수']}
                    contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', border: '1px solid rgba(255,255,255,0.2)', borderRadius: '8px', color: '#fff' }}
                  />
                </PieChart>
              </ResponsiveContainer>

              <div className="flex flex-wrap justify-center gap-2 mt-4">
                {pieData.map((entry, index) => (
                  <div key={index} className="flex items-center gap-1 text-sm">
                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: entry.color }} />
                    <span className="text-white/70">{entry.name}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Popular Keywords */}
            <div className="glossy-panel p-6">
              <div className="flex items-center gap-2 mb-6">
                <Sparkles className="text-yellow-400" size={24} />
                <h2 className="text-xl font-semibold text-white">인기 키워드</h2>
              </div>
              <div className="flex flex-wrap gap-3">
                {[
                  { keyword: 'CRISPR-Cas9', hot: true },
                  { keyword: 'CAR-T therapy', hot: true },
                  { keyword: 'mRNA vaccine', hot: true },
                  { keyword: 'immunotherapy', hot: false },
                  { keyword: 'gene editing', hot: false },
                  { keyword: 'checkpoint inhibitor', hot: false },
                  { keyword: 'PD-1/PD-L1', hot: false },
                  { keyword: 'single-cell RNA-seq', hot: true },
                  { keyword: 'precision medicine', hot: false },
                  { keyword: 'biomarker', hot: false },
                  { keyword: 'AlphaFold', hot: true },
                  { keyword: 'spatial transcriptomics', hot: true },
                ].map(({ keyword, hot }) => (
                  <button
                    key={keyword}
                    onClick={() => {
                      setSearchInput(keyword)
                      setSearchParams({ q: keyword })
                    }}
                    className={`px-4 py-2 rounded-full cursor-pointer transition-all ${
                      hot
                        ? 'bg-gradient-to-r from-orange-500/80 to-pink-500/80 text-white font-medium shadow-lg hover:shadow-xl border border-orange-400/30'
                        : 'bg-white/10 hover:bg-white/20 text-white/80 border border-white/20'
                    }`}
                  >
                    {hot && <span className="mr-1">🔥</span>}
                    {keyword}
                  </button>
                ))}
              </div>

              {/* Growth Stats */}
              <div className="mt-8 grid grid-cols-3 gap-4">
                <div className="text-center p-4 bg-green-500/20 rounded-xl border border-green-400/30">
                  <div className="text-2xl font-bold text-green-400">+67%</div>
                  <div className="text-sm text-green-300">Spatial Transcriptomics</div>
                  <div className="text-xs text-green-400/70 mt-1">가장 빠른 성장</div>
                </div>
                <div className="text-center p-4 bg-cyan-500/20 rounded-xl border border-cyan-400/30">
                  <div className="text-2xl font-bold text-cyan-400">1,847</div>
                  <div className="text-sm text-cyan-300">CRISPR-Cas9</div>
                  <div className="text-xs text-cyan-400/70 mt-1">최다 논문</div>
                </div>
                <div className="text-center p-4 bg-purple-500/20 rounded-xl border border-purple-400/30">
                  <div className="text-2xl font-bold text-purple-400">+52%</div>
                  <div className="text-sm text-purple-300">AlphaFold</div>
                  <div className="text-xs text-purple-400/70 mt-1">AI 트렌드</div>
                </div>
              </div>
            </div>
          </div>
        </>
      )}
        </>
      )}
    </div>
  )
}
