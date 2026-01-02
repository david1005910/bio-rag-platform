import { describe, it, expect, beforeEach, vi } from 'vitest'
import { useAuthStore } from './authStore'

describe('authStore', () => {
  beforeEach(() => {
    // Reset store state before each test
    useAuthStore.setState({
      user: null,
      isAuthenticated: false,
    })
    // Clear localStorage mock
    vi.clearAllMocks()
  })

  describe('initial state', () => {
    it('should have null user initially', () => {
      const { user } = useAuthStore.getState()
      expect(user).toBeNull()
    })

    it('should not be authenticated initially', () => {
      const { isAuthenticated } = useAuthStore.getState()
      expect(isAuthenticated).toBe(false)
    })
  })

  describe('setUser', () => {
    it('should set user and authenticate when user is provided', () => {
      const mockUser = {
        id: '1',
        email: 'test@example.com',
        name: 'Test User',
      }

      useAuthStore.getState().setUser(mockUser)

      const { user, isAuthenticated } = useAuthStore.getState()
      expect(user).toEqual(mockUser)
      expect(isAuthenticated).toBe(true)
    })

    it('should clear user and deauthenticate when null is provided', () => {
      // First set a user
      useAuthStore.getState().setUser({ id: '1', email: 'test@example.com', name: 'Test' })

      // Then clear it
      useAuthStore.getState().setUser(null)

      const { user, isAuthenticated } = useAuthStore.getState()
      expect(user).toBeNull()
      expect(isAuthenticated).toBe(false)
    })
  })

  describe('setToken', () => {
    it('should store token and authenticate when token is provided', () => {
      useAuthStore.getState().setToken('test-token')

      expect(localStorage.setItem).toHaveBeenCalledWith('accessToken', 'test-token')
      expect(useAuthStore.getState().isAuthenticated).toBe(true)
    })

    it('should remove token and deauthenticate when null is provided', () => {
      // First set a token
      useAuthStore.getState().setToken('test-token')

      // Then clear it
      useAuthStore.getState().setToken(null)

      expect(localStorage.removeItem).toHaveBeenCalledWith('accessToken')
      expect(useAuthStore.getState().isAuthenticated).toBe(false)
      expect(useAuthStore.getState().user).toBeNull()
    })
  })

  describe('logout', () => {
    it('should clear user, token and deauthenticate', () => {
      // Setup authenticated state
      useAuthStore.getState().setUser({ id: '1', email: 'test@example.com', name: 'Test' })
      useAuthStore.getState().setToken('test-token')

      // Logout
      useAuthStore.getState().logout()

      expect(localStorage.removeItem).toHaveBeenCalledWith('accessToken')
      expect(useAuthStore.getState().user).toBeNull()
      expect(useAuthStore.getState().isAuthenticated).toBe(false)
    })
  })
})
