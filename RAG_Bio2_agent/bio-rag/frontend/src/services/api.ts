import axios from 'axios'
import type {
  SearchResponse,
  Paper,
  ChatQueryResponse,
  ChatSession,
  ChatMessage,
  AuthTokens,
  User,
  SavedPaper,
  HotTopic,
  KeywordTrend,
} from '@/types'

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api/v1'

// Create axios instance
export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor for auth token
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('accessToken')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('accessToken')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

// ============== Auth API ==============

export const authApi = {
  login: async (email: string, password: string): Promise<AuthTokens> => {
    const response = await api.post('/auth/login', { email, password })
    return response.data
  },

  register: async (
    email: string,
    password: string,
    name: string,
    researchField?: string
  ): Promise<AuthTokens> => {
    const response = await api.post('/auth/register', {
      email,
      password,
      name,
      research_field: researchField,
    })
    return response.data
  },

  logout: async (): Promise<void> => {
    await api.post('/auth/logout')
    localStorage.removeItem('accessToken')
  },

  getMe: async (): Promise<User> => {
    const response = await api.get('/auth/me')
    return response.data
  },
}

// ============== Search API ==============

export const searchApi = {
  search: async (
    query: string,
    limit: number = 10,
    filters?: Record<string, unknown>
  ): Promise<SearchResponse> => {
    const response = await api.post('/search', { query, limit, filters })
    return {
      total: response.data.total,
      tookMs: response.data.took_ms,
      results: response.data.results.map((r: Record<string, unknown>) => ({
        pmid: r.pmid,
        title: r.title,
        abstract: r.abstract,
        relevanceScore: r.relevance_score,
        authors: r.authors,
        journal: r.journal,
        publicationDate: r.publication_date,
        keywords: r.keywords,
      })),
    }
  },

  getPaper: async (pmid: string): Promise<Paper> => {
    const response = await api.get(`/papers/${pmid}`)
    return response.data
  },

  getSimilarPapers: async (pmid: string, limit: number = 5): Promise<Paper[]> => {
    const response = await api.get(`/papers/${pmid}/similar`, { params: { limit } })
    // API returns array directly
    return response.data || []
  },
}

// ============== Chat API ==============

export const chatApi = {
  query: async (
    question: string,
    sessionId?: string,
    contextPmids?: string[]
  ): Promise<ChatQueryResponse> => {
    const response = await api.post('/chat/query', {
      question,
      session_id: sessionId,
      context_pmids: contextPmids,
    })
    return {
      answer: response.data.answer,
      sources: response.data.sources,
      confidence: response.data.confidence,
      processingTimeMs: response.data.processing_time_ms,
      sessionId: response.data.session_id,
    }
  },

  getSessions: async (): Promise<ChatSession[]> => {
    const response = await api.get('/chat/sessions')
    return response.data.sessions
  },

  getSessionMessages: async (sessionId: string): Promise<ChatMessage[]> => {
    const response = await api.get(`/chat/sessions/${sessionId}/messages`)
    return response.data.messages
  },

  deleteSession: async (sessionId: string): Promise<void> => {
    await api.delete(`/chat/sessions/${sessionId}`)
  },
}

// ============== Library API ==============

export const libraryApi = {
  getSavedPapers: async (tag?: string): Promise<SavedPaper[]> => {
    const response = await api.get('/library/papers', { params: { tag } })
    return response.data.papers
  },

  savePaper: async (
    pmid: string,
    tags?: string[],
    notes?: string
  ): Promise<SavedPaper> => {
    const response = await api.post('/library/papers', { pmid, tags, notes })
    return response.data
  },

  deleteSavedPaper: async (paperId: string): Promise<void> => {
    await api.delete(`/library/papers/${paperId}`)
  },

  getTags: async (): Promise<string[]> => {
    const response = await api.get('/library/tags')
    return response.data.tags
  },
}

// ============== Trends API ==============

export const trendsApi = {
  getKeywordTrends: async (keywords: string[]): Promise<KeywordTrend[]> => {
    const params = new URLSearchParams()
    keywords.forEach(k => params.append('keywords', k))
    const response = await api.get(`/trends/keywords?${params.toString()}`)
    return response.data.data
  },

  getHotTopics: async (limit: number = 10): Promise<HotTopic[]> => {
    const response = await api.get('/trends/hot', { params: { limit } })
    return response.data.topics
  },
}
