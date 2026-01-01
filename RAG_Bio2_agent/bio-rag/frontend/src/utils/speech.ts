/**
 * Speech Synthesis Utility
 * 음성 합성 유틸리티 - Yuna(한국어), Shelley(영어) 음성 지원
 */

/**
 * 음성 목록 가져오기 (동기)
 */
const getVoices = (): SpeechSynthesisVoice[] => {
  if (typeof window === 'undefined' || !window.speechSynthesis) {
    return []
  }
  return window.speechSynthesis.getVoices()
}

/**
 * 한국어 Yuna 음성 찾기
 */
const findKoreanVoice = (voices: SpeechSynthesisVoice[]): SpeechSynthesisVoice | undefined => {
  // Yuna 우선
  return voices.find(v => v.name.toLowerCase().includes('yuna')) ||
    voices.find(v => v.lang.startsWith('ko'))
}

/**
 * 영어 Shelley 음성 찾기
 */
const findEnglishVoice = (voices: SpeechSynthesisVoice[]): SpeechSynthesisVoice | undefined => {
  // Shelley 우선
  return voices.find(v => v.name.toLowerCase().includes('shelley') && v.lang.startsWith('en')) ||
    voices.find(v => v.lang.startsWith('en'))
}

/**
 * 음성 목록 미리 로드
 */
export const loadVoices = (): void => {
  if (typeof window === 'undefined' || !window.speechSynthesis) return

  // 초기 로드 시도
  getVoices()

  // voiceschanged 이벤트 등록
  if (window.speechSynthesis.onvoiceschanged === null) {
    window.speechSynthesis.onvoiceschanged = () => {
      const voices = getVoices()
      console.log('Voices loaded:', voices.length, voices.slice(0, 5).map(v => v.name))
    }
  }
}

/**
 * 텍스트 음성 합성
 */
export const speakText = (
  text: string,
  options: {
    lang?: 'ko' | 'en'
    rate?: number
    pitch?: number
    onStart?: () => void
    onEnd?: () => void
    onError?: (error: Error) => void
  } = {}
): void => {
  if (typeof window === 'undefined' || !window.speechSynthesis) {
    console.error('Speech synthesis not supported')
    options.onError?.(new Error('Speech synthesis not supported'))
    return
  }

  const {
    lang = 'ko',
    rate = 1.0,
    pitch = 1.1,
    onStart,
    onEnd,
    onError
  } = options

  // 이전 음성 중지
  window.speechSynthesis.cancel()

  // utterance 생성
  const utterance = new SpeechSynthesisUtterance(text)
  utterance.lang = lang === 'ko' ? 'ko-KR' : 'en-US'
  utterance.rate = rate
  utterance.pitch = pitch
  utterance.volume = 1.0

  // 음성 선택 함수
  const selectVoice = () => {
    const voices = getVoices()
    console.log('Available voices:', voices.length)

    if (voices.length === 0) {
      console.warn('No voices available yet')
      return
    }

    const selectedVoice = lang === 'ko'
      ? findKoreanVoice(voices)
      : findEnglishVoice(voices)

    if (selectedVoice) {
      utterance.voice = selectedVoice
      console.log('Selected voice:', selectedVoice.name)
    } else {
      console.warn('Voice not found for lang:', lang)
    }
  }

  // 이벤트 핸들러
  utterance.onstart = () => {
    console.log('Speech started')
    onStart?.()
  }

  utterance.onend = () => {
    console.log('Speech ended')
    onEnd?.()
  }

  utterance.onerror = (e) => {
    console.error('Speech error:', e.error)
    onError?.(new Error(e.error))
  }

  // 음성 선택 시도
  selectVoice()

  // 음성이 로드되지 않은 경우 재시도
  const voices = getVoices()
  if (voices.length === 0) {
    console.log('Waiting for voices to load...')
    window.speechSynthesis.onvoiceschanged = () => {
      selectVoice()
      window.speechSynthesis.speak(utterance)
    }
  } else {
    // 즉시 재생
    window.speechSynthesis.speak(utterance)
  }
}

/**
 * 음성 중지
 */
export const stopSpeech = (): void => {
  if (typeof window !== 'undefined' && window.speechSynthesis) {
    window.speechSynthesis.cancel()
  }
}

/**
 * 언어 감지 (한국어 비율 체크)
 */
export const detectLanguage = (text: string): 'ko' | 'en' => {
  const koreanRegex = /[가-힣]/g
  const koreanMatches = text.match(koreanRegex) || []
  const totalChars = text.replace(/\s/g, '').length
  const koreanRatio = totalChars > 0 ? koreanMatches.length / totalChars : 0
  return koreanRatio > 0.3 ? 'ko' : 'en'
}

/**
 * 간단한 음성 테스트 (브라우저 콘솔에서 테스트용)
 */
export const testSpeech = (): void => {
  console.log('=== Speech Test ===')
  const voices = getVoices()
  console.log('Total voices:', voices.length)

  const koVoice = findKoreanVoice(voices)
  const enVoice = findEnglishVoice(voices)

  console.log('Korean voice:', koVoice?.name || 'Not found')
  console.log('English voice:', enVoice?.name || 'Not found')

  if (koVoice) {
    speakText('안녕하세요, 음성 테스트입니다.', {
      lang: 'ko',
      onStart: () => console.log('Korean speech started'),
      onEnd: () => console.log('Korean speech ended')
    })
  } else {
    console.error('Korean voice not available')
  }
}

// 브라우저 전역에서 테스트 가능하도록 노출
if (typeof window !== 'undefined') {
  (window as unknown as { testSpeech: typeof testSpeech }).testSpeech = testSpeech
}
