import { describe, it, expect } from 'vitest'
import {
  sanitizeInput,
  validateEmail,
  validateSearchQuery,
  validateChatMessage,
  validateYearParam,
  validateStringArrayParam,
  validateNumberRange,
} from './validation'

describe('validation utilities', () => {
  describe('sanitizeInput', () => {
    it('should trim whitespace', () => {
      expect(sanitizeInput('  hello  ')).toBe('hello')
    })

    it('should limit length to maxLength', () => {
      const longString = 'a'.repeat(600)
      expect(sanitizeInput(longString, 500).length).toBe(500)
    })

    it('should remove script tags', () => {
      const input = 'Hello <script>alert("xss")</script> World'
      expect(sanitizeInput(input)).toBe('Hello  World')
    })

    it('should remove HTML tags', () => {
      const input = 'Hello <b>bold</b> and <i>italic</i>'
      expect(sanitizeInput(input)).toBe('Hello bold and italic')
    })

    it('should handle empty string', () => {
      expect(sanitizeInput('')).toBe('')
    })
  })

  describe('validateEmail', () => {
    it('should return true for valid email', () => {
      expect(validateEmail('test@example.com')).toBe(true)
    })

    it('should return true for email with subdomain', () => {
      expect(validateEmail('test@mail.example.com')).toBe(true)
    })

    it('should return false for email without @', () => {
      expect(validateEmail('testexample.com')).toBe(false)
    })

    it('should return false for email without domain', () => {
      expect(validateEmail('test@')).toBe(false)
    })

    it('should return false for email without local part', () => {
      expect(validateEmail('@example.com')).toBe(false)
    })

    it('should return false for very long email', () => {
      const longEmail = 'a'.repeat(250) + '@example.com'
      expect(validateEmail(longEmail)).toBe(false)
    })
  })

  describe('validateSearchQuery', () => {
    it('should return valid for normal query', () => {
      const result = validateSearchQuery('CRISPR gene editing')
      expect(result.valid).toBe(true)
      expect(result.sanitized).toBe('CRISPR gene editing')
    })

    it('should return invalid for empty query', () => {
      const result = validateSearchQuery('')
      expect(result.valid).toBe(false)
      expect(result.error).toBeDefined()
    })

    it('should return invalid for whitespace-only query', () => {
      const result = validateSearchQuery('   ')
      expect(result.valid).toBe(false)
    })

    it('should return invalid for single character query', () => {
      const result = validateSearchQuery('a')
      expect(result.valid).toBe(false)
    })

    it('should sanitize query with HTML', () => {
      const result = validateSearchQuery('<b>bold</b> query')
      expect(result.valid).toBe(true)
      expect(result.sanitized).toBe('bold query')
    })
  })

  describe('validateChatMessage', () => {
    it('should return valid for normal message', () => {
      const result = validateChatMessage('What is CRISPR?')
      expect(result.valid).toBe(true)
      expect(result.sanitized).toBe('What is CRISPR?')
    })

    it('should return invalid for empty message', () => {
      const result = validateChatMessage('')
      expect(result.valid).toBe(false)
      expect(result.error).toBeDefined()
    })

    it('should return invalid for whitespace-only message', () => {
      const result = validateChatMessage('   ')
      expect(result.valid).toBe(false)
    })

    it('should sanitize message with script tags', () => {
      const result = validateChatMessage('Hello <script>alert(1)</script>')
      expect(result.valid).toBe(true)
      expect(result.sanitized).toBe('Hello ')
    })
  })

  describe('validateYearParam', () => {
    it('should return undefined for null input', () => {
      expect(validateYearParam(null)).toBeUndefined()
    })

    it('should return parsed year for valid input', () => {
      expect(validateYearParam('2024')).toBe(2024)
    })

    it('should return undefined for non-numeric input', () => {
      expect(validateYearParam('abc')).toBeUndefined()
    })

    it('should return undefined for year before 1900', () => {
      expect(validateYearParam('1800')).toBeUndefined()
    })

    it('should return undefined for future year', () => {
      const futureYear = (new Date().getFullYear() + 10).toString()
      expect(validateYearParam(futureYear)).toBeUndefined()
    })

    it('should accept current year', () => {
      const currentYear = new Date().getFullYear()
      expect(validateYearParam(currentYear.toString())).toBe(currentYear)
    })
  })

  describe('validateStringArrayParam', () => {
    it('should return undefined for null input', () => {
      expect(validateStringArrayParam(null)).toBeUndefined()
    })

    it('should parse comma-separated values', () => {
      const result = validateStringArrayParam('a,b,c')
      expect(result).toEqual(['a', 'b', 'c'])
    })

    it('should trim whitespace from items', () => {
      const result = validateStringArrayParam(' a , b , c ')
      expect(result).toEqual(['a', 'b', 'c'])
    })

    it('should filter empty items', () => {
      const result = validateStringArrayParam('a,,b,')
      expect(result).toEqual(['a', 'b'])
    })

    it('should limit number of items', () => {
      const result = validateStringArrayParam('a,b,c,d,e', 3)
      expect(result).toEqual(['a', 'b', 'c'])
    })

    it('should return undefined for empty string', () => {
      expect(validateStringArrayParam('')).toBeUndefined()
    })
  })

  describe('validateNumberRange', () => {
    it('should return true for undefined value', () => {
      expect(validateNumberRange(undefined, 0, 100)).toBe(true)
    })

    it('should return true for value within range', () => {
      expect(validateNumberRange(50, 0, 100)).toBe(true)
    })

    it('should return true for value at min boundary', () => {
      expect(validateNumberRange(0, 0, 100)).toBe(true)
    })

    it('should return true for value at max boundary', () => {
      expect(validateNumberRange(100, 0, 100)).toBe(true)
    })

    it('should return false for value below min', () => {
      expect(validateNumberRange(-1, 0, 100)).toBe(false)
    })

    it('should return false for value above max', () => {
      expect(validateNumberRange(101, 0, 100)).toBe(false)
    })
  })
})
