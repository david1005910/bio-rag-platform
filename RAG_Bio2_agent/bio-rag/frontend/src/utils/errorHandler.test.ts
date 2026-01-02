import { describe, it, expect, vi } from 'vitest'
import { handleApiError, getErrorMessage, logError } from './errorHandler'
import axios, { AxiosError } from 'axios'

// Mock axios.isAxiosError
vi.mock('axios', async () => {
  const actual = await vi.importActual('axios')
  return {
    ...actual,
    default: {
      ...actual,
      isAxiosError: vi.fn(),
    },
  }
})

describe('errorHandler utilities', () => {
  describe('handleApiError', () => {
    it('should return the user message', () => {
      const error = new Error('Something went wrong')
      const result = handleApiError(error, 'User friendly message')
      expect(result).toBe('User friendly message')
    })

    it('should handle null error', () => {
      const result = handleApiError(null, 'Fallback message')
      expect(result).toBe('Fallback message')
    })

    it('should handle undefined error', () => {
      const result = handleApiError(undefined, 'Fallback message')
      expect(result).toBe('Fallback message')
    })
  })

  describe('getErrorMessage', () => {
    beforeEach(() => {
      vi.mocked(axios.isAxiosError).mockReset()
    })

    it('should return 400 message for bad request', () => {
      vi.mocked(axios.isAxiosError).mockReturnValue(true)
      const error = { response: { status: 400 } } as AxiosError
      expect(getErrorMessage(error)).toBe('잘못된 요청입니다')
    })

    it('should return 401 message for unauthorized', () => {
      vi.mocked(axios.isAxiosError).mockReturnValue(true)
      const error = { response: { status: 401 } } as AxiosError
      expect(getErrorMessage(error)).toBe('인증이 필요합니다. 다시 로그인해주세요')
    })

    it('should return 403 message for forbidden', () => {
      vi.mocked(axios.isAxiosError).mockReturnValue(true)
      const error = { response: { status: 403 } } as AxiosError
      expect(getErrorMessage(error)).toBe('접근 권한이 없습니다')
    })

    it('should return 404 message for not found', () => {
      vi.mocked(axios.isAxiosError).mockReturnValue(true)
      const error = { response: { status: 404 } } as AxiosError
      expect(getErrorMessage(error)).toBe('요청한 리소스를 찾을 수 없습니다')
    })

    it('should return 429 message for rate limit', () => {
      vi.mocked(axios.isAxiosError).mockReturnValue(true)
      const error = { response: { status: 429 } } as AxiosError
      expect(getErrorMessage(error)).toBe('너무 많은 요청이 발생했습니다. 잠시 후 다시 시도하세요')
    })

    it('should return 500 message for server error', () => {
      vi.mocked(axios.isAxiosError).mockReturnValue(true)
      const error = { response: { status: 500 } } as AxiosError
      expect(getErrorMessage(error)).toBe('서버 오류가 발생했습니다. 잠시 후 다시 시도하세요')
    })

    it('should return connection message for 502/503/504', () => {
      vi.mocked(axios.isAxiosError).mockReturnValue(true)

      const error502 = { response: { status: 502 } } as AxiosError
      expect(getErrorMessage(error502)).toBe('서버에 연결할 수 없습니다. 잠시 후 다시 시도하세요')

      const error503 = { response: { status: 503 } } as AxiosError
      expect(getErrorMessage(error503)).toBe('서버에 연결할 수 없습니다. 잠시 후 다시 시도하세요')

      const error504 = { response: { status: 504 } } as AxiosError
      expect(getErrorMessage(error504)).toBe('서버에 연결할 수 없습니다. 잠시 후 다시 시도하세요')
    })

    it('should return unknown error message for other status codes', () => {
      vi.mocked(axios.isAxiosError).mockReturnValue(true)
      const error = { response: { status: 418 } } as AxiosError
      expect(getErrorMessage(error)).toBe('알 수 없는 오류가 발생했습니다')
    })

    it('should return network error message', () => {
      vi.mocked(axios.isAxiosError).mockReturnValue(false)
      const error = new Error('Network Error')
      expect(getErrorMessage(error)).toBe('네트워크 연결을 확인하세요')
    })

    it('should return timeout message', () => {
      vi.mocked(axios.isAxiosError).mockReturnValue(false)
      const error = new Error('Request timeout exceeded')
      expect(getErrorMessage(error)).toBe('요청 시간이 초과되었습니다. 다시 시도하세요')
    })

    it('should return generic message for unknown errors', () => {
      vi.mocked(axios.isAxiosError).mockReturnValue(false)
      expect(getErrorMessage('unknown')).toBe('요청 처리 중 오류가 발생했습니다')
    })
  })

  describe('logError', () => {
    it('should not throw when called', () => {
      expect(() => logError('TestContext', new Error('Test'))).not.toThrow()
    })

    it('should handle null error', () => {
      expect(() => logError('TestContext', null)).not.toThrow()
    })

    it('should handle string error', () => {
      expect(() => logError('TestContext', 'string error')).not.toThrow()
    })
  })
})
