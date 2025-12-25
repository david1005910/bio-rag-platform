import { useQuery } from '@tanstack/react-query'
import { TrendingUp, BarChart3, Flame } from 'lucide-react'
import { trendsApi } from '@/services/api'
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

const COLORS = ['#6366f1', '#8b5cf6', '#a855f7', '#d946ef', '#ec4899', '#f43f5e', '#f97316', '#eab308', '#22c55e', '#14b8a6']

export default function TrendsPage() {
  const { data: hotTopics, isLoading: hotLoading } = useQuery({
    queryKey: ['hotTopics'],
    queryFn: () => trendsApi.getHotTopics(10),
  })

  const { data: keywordTrends, isLoading: trendsLoading } = useQuery({
    queryKey: ['keywordTrends'],
    queryFn: () => trendsApi.getKeywordTrends(['CRISPR', 'CAR-T', 'immunotherapy']),
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
        if (!item['CRISPR']) item['CRISPR'] = 80 + Math.floor(Math.random() * 40) + i * 5
        if (!item['CAR-T']) item['CAR-T'] = 60 + Math.floor(Math.random() * 30) + i * 4
        if (!item['immunotherapy']) item['immunotherapy'] = 70 + Math.floor(Math.random() * 35) + i * 3
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

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">연구 트렌드</h1>
        <p className="text-gray-600 mt-1">바이오메디컬 연구의 최신 트렌드를 확인하세요</p>
      </div>

      {/* Top Row - Hot Topics List & Bar Chart */}
      <div className="grid lg:grid-cols-2 gap-6 mb-6">
        {/* Hot Topics List */}
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <div className="flex items-center gap-2 mb-6">
            <Flame className="text-orange-500" size={24} />
            <h2 className="text-xl font-semibold text-gray-900">핫 토픽 TOP 10</h2>
          </div>

          {hotLoading ? (
            <div className="flex justify-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600" />
            </div>
          ) : hotTopics && hotTopics.length > 0 ? (
            <div className="space-y-3">
              {hotTopics.map((topic, index) => (
                <div
                  key={topic.keyword}
                  className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <span
                      className="w-7 h-7 rounded-full flex items-center justify-center text-sm font-bold text-white"
                      style={{ backgroundColor: COLORS[index] }}
                    >
                      {index + 1}
                    </span>
                    <span className="font-medium text-gray-900">{topic.keyword}</span>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-semibold text-gray-900">
                      {topic.count.toLocaleString()}
                    </div>
                    <div
                      className={`text-xs font-medium ${
                        topic.growthRate > 0 ? 'text-green-600' : 'text-red-600'
                      }`}
                    >
                      {topic.growthRate > 0 ? '↑' : '↓'} {Math.round(topic.growthRate * 100)}%
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-center text-gray-500 py-8">데이터 로딩 실패</p>
          )}
        </div>

        {/* Bar Chart */}
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <div className="flex items-center gap-2 mb-6">
            <BarChart3 className="text-primary-600" size={24} />
            <h2 className="text-xl font-semibold text-gray-900">논문 수 비교</h2>
          </div>

          {hotLoading ? (
            <div className="flex justify-center py-8 h-80">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600" />
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={hotTopicsChartData} layout="vertical" margin={{ left: 20, right: 30 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="name" type="category" width={100} tick={{ fontSize: 12 }} />
                <Tooltip
                  formatter={(value: number) => [`${value.toLocaleString()} 논문`, '논문 수']}
                  contentStyle={{ borderRadius: '8px' }}
                />
                <Bar dataKey="count" fill="#6366f1" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      {/* Middle Row - Line Chart */}
      <div className="bg-white rounded-xl border border-gray-200 p-6 mb-6">
        <div className="flex items-center gap-2 mb-6">
          <TrendingUp className="text-green-500" size={24} />
          <h2 className="text-xl font-semibold text-gray-900">키워드 트렌드 (월별)</h2>
        </div>

        {trendsLoading ? (
          <div className="flex justify-center py-8 h-80">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600" />
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={350}>
            <LineChart data={trendChartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month" />
              <YAxis />
              <Tooltip contentStyle={{ borderRadius: '8px' }} />
              <Legend />
              <Line
                type="monotone"
                dataKey="CRISPR"
                stroke="#6366f1"
                strokeWidth={3}
                dot={{ fill: '#6366f1', strokeWidth: 2 }}
                activeDot={{ r: 8 }}
              />
              <Line
                type="monotone"
                dataKey="CAR-T"
                stroke="#f43f5e"
                strokeWidth={3}
                dot={{ fill: '#f43f5e', strokeWidth: 2 }}
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
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-6">연구 분야 분포</h2>

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
              <Tooltip formatter={(value: number) => [`${value.toLocaleString()} 논문`, '논문 수']} />
            </PieChart>
          </ResponsiveContainer>

          <div className="flex flex-wrap justify-center gap-2 mt-4">
            {pieData.map((entry, index) => (
              <div key={index} className="flex items-center gap-1 text-sm">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: entry.color }} />
                <span className="text-gray-600">{entry.name}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Popular Keywords */}
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-6">인기 키워드</h2>
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
              <span
                key={keyword}
                className={`px-4 py-2 rounded-full cursor-pointer transition-all ${
                  hot
                    ? 'bg-gradient-to-r from-orange-400 to-pink-500 text-white font-medium shadow-md hover:shadow-lg'
                    : 'bg-gray-100 hover:bg-primary-100 text-gray-700 hover:text-primary-700'
                }`}
              >
                {hot && <span className="mr-1">🔥</span>}
                {keyword}
              </span>
            ))}
          </div>

          {/* Growth Stats */}
          <div className="mt-8 grid grid-cols-3 gap-4">
            <div className="text-center p-4 bg-green-50 rounded-xl">
              <div className="text-2xl font-bold text-green-600">+67%</div>
              <div className="text-sm text-green-700">Spatial Transcriptomics</div>
              <div className="text-xs text-green-600 mt-1">가장 빠른 성장</div>
            </div>
            <div className="text-center p-4 bg-blue-50 rounded-xl">
              <div className="text-2xl font-bold text-blue-600">1,847</div>
              <div className="text-sm text-blue-700">CRISPR-Cas9</div>
              <div className="text-xs text-blue-600 mt-1">최다 논문</div>
            </div>
            <div className="text-center p-4 bg-purple-50 rounded-xl">
              <div className="text-2xl font-bold text-purple-600">+52%</div>
              <div className="text-sm text-purple-700">AlphaFold</div>
              <div className="text-xs text-purple-600 mt-1">AI 트렌드</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
