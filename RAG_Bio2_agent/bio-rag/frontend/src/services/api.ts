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
    return {
      accessToken: response.data.access_token,
      tokenType: response.data.token_type,
      expiresIn: response.data.expires_in,
    }
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
    return {
      accessToken: response.data.access_token,
      tokenType: response.data.token_type,
      expiresIn: response.data.expires_in,
    }
  },

  logout: async (): Promise<void> => {
    await api.post('/auth/logout')
    localStorage.removeItem('accessToken')
  },

  getMe: async (): Promise<User> => {
    const response = await api.get('/auth/me')
    return {
      id: response.data.id,
      email: response.data.email,
      name: response.data.name,
      researchField: response.data.research_field,
    }
  },
}

// ============== Search API ==============

export const searchApi = {
  search: async (
    query: string,
    limit: number = 10,
    filters?: Record<string, unknown>,
    source: 'pubmed' | 'mock' = 'pubmed'
  ): Promise<SearchResponse> => {
    const response = await api.post('/search', { query, limit, filters, source })
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
    const r = response.data
    return {
      pmid: r.pmid,
      title: r.title,
      abstract: r.abstract,
      authors: r.authors || [],
      journal: r.journal,
      publicationDate: r.publication_date,
      doi: r.doi,
      keywords: r.keywords || [],
      meshTerms: r.mesh_terms || [],
    }
  },

  getSimilarPapers: async (pmid: string, limit: number = 5): Promise<Paper[]> => {
    const response = await api.get(`/papers/${pmid}/similar`, { params: { limit } })
    // API returns array directly
    return response.data || []
  },

  summarize: async (text: string, language: string = 'ko'): Promise<{ summary: string }> => {
    const response = await api.post('/summarize', { text, language })
    return {
      summary: response.data.summary,
    }
  },

  translate: async (text: string, sourceLang: string = 'ko', targetLang: string = 'en'): Promise<{ original: string; translated: string }> => {
    const response = await api.post('/translate', {
      text,
      source_lang: sourceLang,
      target_lang: targetLang,
    })
    return {
      original: response.data.original,
      translated: response.data.translated,
    }
  },
}

// ============== Chat API ==============

export interface ChatQueryOptions {
  question: string
  sessionId?: string
  contextPmids?: string[]
  useVectordb?: boolean
  searchMode?: 'hybrid' | 'dense' | 'sparse'
  denseWeight?: number
}

export interface ChatQueryResponseExtended extends ChatQueryResponse {
  vectordbUsed: boolean
  searchMode?: string
}

export const chatApi = {
  query: async (
    question: string,
    sessionId?: string,
    contextPmids?: string[],
    options?: {
      useVectordb?: boolean
      searchMode?: 'hybrid' | 'dense' | 'sparse'
      denseWeight?: number
    }
  ): Promise<ChatQueryResponseExtended> => {
    const response = await api.post('/chat/query', {
      question,
      session_id: sessionId,
      context_pmids: contextPmids,
      use_vectordb: options?.useVectordb ?? true,
      search_mode: options?.searchMode ?? 'hybrid',
      dense_weight: options?.denseWeight ?? 0.7,
    })
    return {
      answer: response.data.answer,
      sources: response.data.sources.map((s: any) => ({
        ...s,
        sourceType: s.source_type,
        denseScore: s.dense_score,
        sparseScore: s.sparse_score,
      })),
      confidence: response.data.confidence,
      processingTimeMs: response.data.processing_time_ms,
      sessionId: response.data.session_id,
      vectordbUsed: response.data.vectordb_used,
      searchMode: response.data.search_mode,
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

// ============== VectorDB API ==============

export interface PaperForVectorDB {
  pmid: string
  title: string
  abstract: string
  authors: string[]
  journal: string
  publication_date?: string
  keywords: string[]
}

export interface SavePapersResponse {
  saved_count: number
  total_chunks: number
  processing_time_ms: number
  paper_ids: string[]
}

export interface VectorDBStats {
  collection_name: string
  vectors_count: number
  status: string
}

export interface VectorSearchResult {
  pmid: string
  title: string
  text: string
  score: number
  dense_score?: number
  sparse_score?: number
  section: string
}

export interface VectorSearchResponse {
  results: VectorSearchResult[]
  took_ms: number
  search_mode: string
}

export const vectordbApi = {
  savePapers: async (papers: PaperForVectorDB[]): Promise<SavePapersResponse> => {
    const response = await api.post('/vectordb/papers/save', { papers })
    return response.data
  },

  getStats: async (): Promise<VectorDBStats> => {
    const response = await api.get('/vectordb/stats')
    return response.data
  },

  search: async (
    query: string,
    topK: number = 5,
    searchMode: 'hybrid' | 'dense' | 'sparse' = 'hybrid',
    denseWeight: number = 0.7
  ): Promise<VectorSearchResponse> => {
    const response = await api.post('/vectordb/search', {
      query,
      top_k: topK,
      search_mode: searchMode,
      dense_weight: denseWeight
    })
    return response.data
  },

  clear: async (): Promise<void> => {
    await api.delete('/vectordb/clear')
  },
}
