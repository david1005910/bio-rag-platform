import { describe, it, expect, beforeEach } from 'vitest'
import { useSearchStore } from './searchStore'

describe('searchStore', () => {
  beforeEach(() => {
    // Reset store state before each test
    useSearchStore.setState({
      lastQuery: '',
      lastResults: null,
      lastFilters: {},
      currentPage: 1,
      autoSavedQuery: null,
    })
  })

  describe('initial state', () => {
    it('should have empty query initially', () => {
      const { lastQuery } = useSearchStore.getState()
      expect(lastQuery).toBe('')
    })

    it('should have null results initially', () => {
      const { lastResults } = useSearchStore.getState()
      expect(lastResults).toBeNull()
    })

    it('should have empty filters initially', () => {
      const { lastFilters } = useSearchStore.getState()
      expect(lastFilters).toEqual({})
    })

    it('should be on page 1 initially', () => {
      const { currentPage } = useSearchStore.getState()
      expect(currentPage).toBe(1)
    })

    it('should have no auto-saved query initially', () => {
      const { autoSavedQuery } = useSearchStore.getState()
      expect(autoSavedQuery).toBeNull()
    })
  })

  describe('setLastSearch', () => {
    it('should update query, results, and filters', () => {
      const mockResults = [
        {
          pmid: '12345',
          title: 'Test Paper',
          abstract: 'Test abstract',
          authors: ['Author 1'],
          journal: 'Test Journal',
          publicationDate: '2024-01-01',
          relevanceScore: 0.95,
          keywords: ['test', 'paper'],
        },
      ]
      const mockFilters = { yearFrom: 2020, yearTo: 2024 }

      useSearchStore.getState().setLastSearch('test query', mockResults, mockFilters)

      const state = useSearchStore.getState()
      expect(state.lastQuery).toBe('test query')
      expect(state.lastResults).toEqual(mockResults)
      expect(state.lastFilters).toEqual(mockFilters)
    })

    it('should reset page to 1 when setting new search', () => {
      // Set page to something other than 1
      useSearchStore.getState().setCurrentPage(5)

      // Set new search
      useSearchStore.getState().setLastSearch('new query', [], {})

      expect(useSearchStore.getState().currentPage).toBe(1)
    })
  })

  describe('setCurrentPage', () => {
    it('should update current page', () => {
      useSearchStore.getState().setCurrentPage(3)
      expect(useSearchStore.getState().currentPage).toBe(3)
    })
  })

  describe('setAutoSavedQuery', () => {
    it('should set auto-saved query', () => {
      useSearchStore.getState().setAutoSavedQuery('saved query')
      expect(useSearchStore.getState().autoSavedQuery).toBe('saved query')
    })

    it('should clear auto-saved query when null is provided', () => {
      useSearchStore.getState().setAutoSavedQuery('saved query')
      useSearchStore.getState().setAutoSavedQuery(null)
      expect(useSearchStore.getState().autoSavedQuery).toBeNull()
    })
  })

  describe('clearSearch', () => {
    it('should reset all state to initial values', () => {
      // Setup some state
      useSearchStore.getState().setLastSearch('test', [{ pmid: '1' } as never], { yearFrom: 2020 })
      useSearchStore.getState().setCurrentPage(5)
      useSearchStore.getState().setAutoSavedQuery('saved')

      // Clear
      useSearchStore.getState().clearSearch()

      const state = useSearchStore.getState()
      expect(state.lastQuery).toBe('')
      expect(state.lastResults).toBeNull()
      expect(state.lastFilters).toEqual({})
      expect(state.currentPage).toBe(1)
      expect(state.autoSavedQuery).toBeNull()
    })
  })
})
