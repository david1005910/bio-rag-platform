import { useEffect, useState } from 'react'

interface DNAHelixProps {
  count?: number
}

export default function DNAHelix({ count = 3 }: DNAHelixProps) {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) return null

  // Generate multiple DNA helices at different positions
  const helixPositions = [
    { left: '5%', top: '10%', scale: 0.8, delay: 0 },
    { right: '8%', top: '20%', scale: 0.6, delay: 2 },
    { left: '15%', bottom: '15%', scale: 0.7, delay: 4 },
    { right: '12%', bottom: '25%', scale: 0.5, delay: 1 },
    { left: '45%', top: '5%', scale: 0.4, delay: 3 },
  ].slice(0, count)

  return (
    <>
      {helixPositions.map((pos, index) => (
        <div
          key={index}
          className="dna-helix-container"
          style={{
            position: 'absolute',
            ...pos,
            transform: `scale(${pos.scale})`,
            opacity: 0.4,
            animationDelay: `${pos.delay}s`,
          }}
        >
          <DNAStrand delay={pos.delay} />
        </div>
      ))}
    </>
  )
}

interface DNAStrandProps {
  delay: number
}

function DNAStrand({ delay }: DNAStrandProps) {
  const basePairs = 12
  const colors = {
    adenine: '#ef4444',    // Red (A)
    thymine: '#22c55e',    // Green (T)
    guanine: '#3b82f6',    // Blue (G)
    cytosine: '#eab308',   // Yellow (C)
  }

  const pairs: Array<{ left: string; right: string }> = []
  for (let i = 0; i < basePairs; i++) {
    // A-T or G-C base pairs
    if (i % 2 === 0) {
      pairs.push({ left: colors.adenine, right: colors.thymine })
    } else {
      pairs.push({ left: colors.guanine, right: colors.cytosine })
    }
  }

  return (
    <div
      className="dna-strand"
      style={{
        width: '80px',
        height: '400px',
        position: 'relative',
        animation: `dna-rotate 20s linear ${delay}s infinite`,
      }}
    >
      {pairs.map((pair, i) => (
        <div
          key={i}
          className="base-pair"
          style={{
            position: 'absolute',
            top: `${(i / basePairs) * 100}%`,
            left: '50%',
            transform: 'translateX(-50%)',
            animation: `dna-wave 4s ease-in-out ${delay + i * 0.2}s infinite`,
          }}
        >
          {/* Left backbone (phosphate) */}
          <div
            className="backbone-left"
            style={{
              position: 'absolute',
              left: '-30px',
              width: '12px',
              height: '12px',
              borderRadius: '50%',
              background: 'linear-gradient(135deg, #06b6d4 0%, #0891b2 100%)',
              boxShadow: '0 0 10px rgba(6, 182, 212, 0.5)',
              animation: `glow-pulse 2s ease-in-out ${i * 0.1}s infinite`,
            }}
          />

          {/* Left base */}
          <div
            style={{
              position: 'absolute',
              left: '-15px',
              width: '20px',
              height: '8px',
              background: pair.left,
              borderRadius: '4px 0 0 4px',
              boxShadow: `0 0 8px ${pair.left}60`,
            }}
          />

          {/* Hydrogen bond (connection) */}
          <div
            style={{
              position: 'absolute',
              left: '5px',
              width: '30px',
              height: '2px',
              background: 'rgba(255, 255, 255, 0.3)',
              boxShadow: '0 0 4px rgba(255, 255, 255, 0.2)',
            }}
          />

          {/* Right base */}
          <div
            style={{
              position: 'absolute',
              right: '-15px',
              width: '20px',
              height: '8px',
              background: pair.right,
              borderRadius: '0 4px 4px 0',
              boxShadow: `0 0 8px ${pair.right}60`,
            }}
          />

          {/* Right backbone (phosphate) */}
          <div
            className="backbone-right"
            style={{
              position: 'absolute',
              right: '-30px',
              width: '12px',
              height: '12px',
              borderRadius: '50%',
              background: 'linear-gradient(135deg, #06b6d4 0%, #0891b2 100%)',
              boxShadow: '0 0 10px rgba(6, 182, 212, 0.5)',
              animation: `glow-pulse 2s ease-in-out ${i * 0.1 + 0.5}s infinite`,
            }}
          />
        </div>
      ))}

      {/* Sugar-phosphate backbone lines */}
      <svg
        style={{
          position: 'absolute',
          width: '100%',
          height: '100%',
          left: 0,
          top: 0,
          pointerEvents: 'none',
        }}
      >
        {/* Left backbone */}
        <path
          d={generateHelixPath('left', basePairs)}
          stroke="url(#backboneGradient)"
          strokeWidth="3"
          fill="none"
          opacity="0.6"
        />
        {/* Right backbone */}
        <path
          d={generateHelixPath('right', basePairs)}
          stroke="url(#backboneGradient)"
          strokeWidth="3"
          fill="none"
          opacity="0.6"
        />
        <defs>
          <linearGradient id="backboneGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#06b6d4" />
            <stop offset="50%" stopColor="#8b5cf6" />
            <stop offset="100%" stopColor="#06b6d4" />
          </linearGradient>
        </defs>
      </svg>
    </div>
  )
}

function generateHelixPath(side: 'left' | 'right', pairs: number): string {
  const points: string[] = []
  const amplitude = 25
  const offset = side === 'left' ? 0 : Math.PI

  for (let i = 0; i <= pairs; i++) {
    const y = (i / pairs) * 380 + 10
    const x = 40 + Math.sin((i / pairs) * Math.PI * 4 + offset) * amplitude
    points.push(`${i === 0 ? 'M' : 'L'} ${x} ${y}`)
  }

  return points.join(' ')
}
