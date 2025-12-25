import { Link } from 'react-router-dom'
import { Search, MessageSquare, TrendingUp, Zap } from 'lucide-react'

const features = [
  {
    icon: Search,
    title: '의미 기반 검색',
    description: '자연어로 논문을 검색하세요. AI가 의미를 이해하고 관련 논문을 찾아드립니다.',
    link: '/search',
  },
  {
    icon: MessageSquare,
    title: 'AI 논문 Q&A',
    description: '연구 질문을 하면 관련 논문을 기반으로 신뢰할 수 있는 답변을 제공합니다.',
    link: '/chat',
  },
  {
    icon: TrendingUp,
    title: '연구 트렌드',
    description: '실시간 연구 트렌드를 파악하고 떠오르는 주제를 확인하세요.',
    link: '/trends',
  },
]

export default function HomePage() {
  return (
    <div className="bg-white">
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24">
          <div className="text-center">
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-gray-900 tracking-tight">
              AI로 더 빠르게,
              <br />
              <span className="text-primary-600">바이오 연구</span>를 혁신하세요
            </h1>
            <p className="mt-6 text-lg sm:text-xl text-gray-600 max-w-3xl mx-auto">
              Bio-RAG는 PubMed 논문을 AI로 분석하여 연구자들이 더 빠르게
              인사이트를 얻을 수 있도록 돕습니다.
            </p>
            <div className="mt-10 flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                to="/search"
                className="inline-flex items-center justify-center px-8 py-4 bg-primary-600 text-white font-medium rounded-xl hover:bg-primary-700 transition-colors"
              >
                <Search className="mr-2" size={20} />
                논문 검색 시작
              </Link>
              <Link
                to="/chat"
                className="inline-flex items-center justify-center px-8 py-4 bg-gray-100 text-gray-900 font-medium rounded-xl hover:bg-gray-200 transition-colors"
              >
                <MessageSquare className="mr-2" size={20} />
                AI에게 질문하기
              </Link>
            </div>
          </div>
        </div>

        {/* Background decoration */}
        <div className="absolute inset-0 -z-10 overflow-hidden">
          <div className="absolute left-1/2 top-0 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] rounded-full bg-gradient-to-r from-primary-100 to-blue-100 blur-3xl opacity-50" />
        </div>
      </div>

      {/* Features Section */}
      <div className="py-24 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-gray-900">
              연구를 더 스마트하게
            </h2>
            <p className="mt-4 text-lg text-gray-600">
              AI 기술로 논문 분석 시간을 70% 단축하세요
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {features.map((feature) => {
              const Icon = feature.icon
              return (
                <Link
                  key={feature.title}
                  to={feature.link}
                  className="group bg-white p-8 rounded-2xl shadow-sm hover:shadow-lg transition-shadow"
                >
                  <div className="w-12 h-12 bg-primary-100 rounded-xl flex items-center justify-center mb-6 group-hover:bg-primary-200 transition-colors">
                    <Icon className="text-primary-600" size={24} />
                  </div>
                  <h3 className="text-xl font-semibold text-gray-900 mb-3">
                    {feature.title}
                  </h3>
                  <p className="text-gray-600">
                    {feature.description}
                  </p>
                </Link>
              )
            })}
          </div>
        </div>
      </div>

      {/* Stats Section */}
      <div className="py-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8 text-center">
            <div>
              <div className="text-4xl font-bold text-primary-600">35M+</div>
              <div className="mt-2 text-gray-600">PubMed 논문</div>
            </div>
            <div>
              <div className="text-4xl font-bold text-primary-600">&lt;2s</div>
              <div className="mt-2 text-gray-600">평균 응답 시간</div>
            </div>
            <div>
              <div className="text-4xl font-bold text-primary-600">95%</div>
              <div className="mt-2 text-gray-600">답변 정확도</div>
            </div>
            <div>
              <div className="text-4xl font-bold text-primary-600">24/7</div>
              <div className="mt-2 text-gray-600">AI 지원</div>
            </div>
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div className="py-24 bg-primary-600">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <Zap className="mx-auto text-primary-200 mb-6" size={48} />
          <h2 className="text-3xl font-bold text-white mb-4">
            지금 바로 시작하세요
          </h2>
          <p className="text-lg text-primary-100 mb-8 max-w-2xl mx-auto">
            무료로 Bio-RAG를 체험하고 연구 효율성을 높이세요.
            회원가입 없이도 기본 기능을 사용할 수 있습니다.
          </p>
          <Link
            to="/register"
            className="inline-flex items-center justify-center px-8 py-4 bg-white text-primary-600 font-medium rounded-xl hover:bg-gray-100 transition-colors"
          >
            무료 회원가입
          </Link>
        </div>
      </div>
    </div>
  )
}
