import { useState, useRef, useEffect } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Send, Loader2, BookOpen, RefreshCw } from 'lucide-react'
import { chatApi } from '@/services/api'
import { useChatStore } from '@/store/chatStore'
import type { ChatMessage, ChatSource } from '@/types'

export default function ChatPage() {
  const [input, setInput] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const {
    messages,
    currentSessionId,
    isLoading,
    addMessage,
    setLoading,
    clearMessages,
  } = useChatStore()

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const queryMutation = useMutation({
    mutationFn: (question: string) =>
      chatApi.query(question, currentSessionId || undefined),
    onMutate: () => {
      setLoading(true)
    },
    onSuccess: (data) => {
      const assistantMessage: ChatMessage = {
        id: Date.now().toString(),
        role: 'assistant',
        content: data.answer,
        sources: data.sources,
        createdAt: new Date().toISOString(),
      }
      addMessage(assistantMessage)
      setLoading(false)
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
    if (!input.trim() || isLoading) return

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
      createdAt: new Date().toISOString(),
    }

    addMessage(userMessage)
    queryMutation.mutate(input.trim())
    setInput('')
  }

  return (
    <div className="flex flex-col h-[calc(100vh-12rem)]">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-4 border-b border-gray-200 bg-white">
        <div>
          <h1 className="text-xl font-bold text-gray-900">AI 논문 Q&A</h1>
          <p className="text-sm text-gray-500">
            바이오메디컬 연구에 관한 질문을 하세요
          </p>
        </div>
        <button
          onClick={clearMessages}
          className="flex items-center gap-2 px-3 py-2 text-sm text-gray-600 hover:bg-gray-100 rounded-lg"
        >
          <RefreshCw size={16} />
          새 대화
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50">
        {messages.length === 0 && (
          <div className="text-center py-16">
            <BookOpen className="mx-auto text-gray-300 mb-4" size={64} />
            <h3 className="text-xl font-medium text-gray-600 mb-2">
              무엇이든 물어보세요
            </h3>
            <p className="text-gray-500 max-w-md mx-auto mb-8">
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
                  className="px-4 py-2 bg-white border border-gray-200 rounded-full text-sm text-gray-600 hover:border-primary-300 hover:text-primary-600 transition-colors"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}

        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-white rounded-2xl rounded-tl-none px-6 py-4 shadow-sm">
              <div className="flex items-center gap-2 text-gray-500">
                <Loader2 className="animate-spin" size={18} />
                답변 생성 중...
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-4 border-t border-gray-200 bg-white">
        <form onSubmit={handleSubmit} className="flex gap-4">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="질문을 입력하세요..."
            className="flex-1 px-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-500"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            className="px-6 py-3 bg-primary-600 text-white rounded-xl hover:bg-primary-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Send size={20} />
          </button>
        </form>
      </div>
    </div>
  )
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === 'user'

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-3xl ${
          isUser
            ? 'bg-primary-600 text-white rounded-2xl rounded-tr-none'
            : 'bg-white text-gray-800 rounded-2xl rounded-tl-none shadow-sm'
        } px-6 py-4`}
      >
        <p className="whitespace-pre-wrap">{message.content}</p>

        {message.sources && message.sources.length > 0 && (
          <div className="mt-4 pt-4 border-t border-gray-200">
            <p className="text-sm font-medium text-gray-700 mb-2">참고 문헌:</p>
            <div className="space-y-2">
              {message.sources.map((source: ChatSource, index: number) => (
                <div
                  key={source.pmid}
                  className="text-sm bg-gray-50 p-3 rounded-lg"
                >
                  <div className="font-medium text-gray-800">
                    [{index + 1}] PMID: {source.pmid}
                  </div>
                  <div className="text-gray-600">{source.title}</div>
                  <div className="text-xs text-gray-400 mt-1">
                    관련도: {Math.round(source.relevance * 100)}%
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
