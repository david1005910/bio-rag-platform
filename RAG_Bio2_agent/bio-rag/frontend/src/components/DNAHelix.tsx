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

  const helixPositions = [
    { left: '3%', top: '5%', scale: 1.2, delay: 0, rotate: -15 },
    { right: '5%', top: '10%', scale: 1.0, delay: 2, rotate: 10 },
    { left: '8%', bottom: '5%', scale: 0.9, delay: 4, rotate: -20 },
    { right: '3%', bottom: '10%', scale: 0.8, delay: 1, rotate: 15 },
  ].slice(0, count)

  return (
    <>
      {helixPositions.map((pos, index) => (
        <div
          key={index}
          className="dna-helix-container"
          style={{
            position: 'absolute',
            left: pos.left,
            right: pos.right,
            top: pos.top,
            bottom: pos.bottom,
            transform: `scale(${pos.scale}) rotate(${pos.rotate}deg)`,
            opacity: 0.6,
            zIndex: 1,
          }}
        >
          <DNA3DHelix delay={pos.delay} />
        </div>
      ))}
    </>
  )
}

interface DNA3DHelixProps {
  delay: number
}

function DNA3DHelix({ delay }: DNA3DHelixProps) {
  const nucleotides = 20 // Number of nucleotide pairs
  const helixHeight = 600
  const helixRadius = 40
  const verticalSpacing = helixHeight / nucleotides

  // Base pair colors
  const basePairs = [
    { left: '#ef4444', right: '#22c55e', label: 'A-T' }, // Adenine-Thymine (Red-Green)
    { left: '#3b82f6', right: '#eab308', label: 'G-C' }, // Guanine-Cytosine (Blue-Yellow)
  ]

  return (
    <div
      className="dna-3d-helix"
      style={{
        width: `${helixRadius * 3}px`,
        height: `${helixHeight}px`,
        position: 'relative',
        transformStyle: 'preserve-3d',
        animation: `dna-spin 15s linear ${delay}s infinite`,
      }}
    >
      {Array.from({ length: nucleotides }).map((_, i) => {
        const angle = (i / nucleotides) * Math.PI * 4 // 2 full rotations
        const y = i * verticalSpacing
        const pair = basePairs[i % 2]

        // Calculate 3D positions for helix
        const leftX = Math.cos(angle) * helixRadius + helixRadius * 1.5
        const leftZ = Math.sin(angle) * helixRadius
        const rightX = Math.cos(angle + Math.PI) * helixRadius + helixRadius * 1.5
        const rightZ = Math.sin(angle + Math.PI) * helixRadius

        // Opacity based on Z position (depth effect)
        const leftOpacity = 0.5 + (leftZ / helixRadius) * 0.5
        const rightOpacity = 0.5 + (rightZ / helixRadius) * 0.5

        return (
          <div key={i} style={{ position: 'absolute', width: '100%', height: '100%' }}>
            {/* Left nucleotide (backbone sphere) */}
            <div
              className="nucleotide-sphere"
              style={{
                position: 'absolute',
                left: `${leftX - 12}px`,
                top: `${y}px`,
                width: '24px',
                height: '24px',
                borderRadius: '50%',
                background: `radial-gradient(circle at 30% 30%, ${pair.left}, ${pair.left}88 60%, ${pair.left}44)`,
                boxShadow: `0 0 20px ${pair.left}80, inset 0 -5px 15px rgba(0,0,0,0.3)`,
                opacity: leftOpacity,
                transform: `translateZ(${leftZ}px)`,
                animation: `nucleotide-pulse 2s ease-in-out ${delay + i * 0.1}s infinite`,
              }}
            />

            {/* Right nucleotide (backbone sphere) */}
            <div
              className="nucleotide-sphere"
              style={{
                position: 'absolute',
                left: `${rightX - 12}px`,
                top: `${y}px`,
                width: '24px',
                height: '24px',
                borderRadius: '50%',
                background: `radial-gradient(circle at 30% 30%, ${pair.right}, ${pair.right}88 60%, ${pair.right}44)`,
                boxShadow: `0 0 20px ${pair.right}80, inset 0 -5px 15px rgba(0,0,0,0.3)`,
                opacity: rightOpacity,
                transform: `translateZ(${rightZ}px)`,
                animation: `nucleotide-pulse 2s ease-in-out ${delay + i * 0.1 + 0.5}s infinite`,
              }}
            />

            {/* Hydrogen bond (connecting line) */}
            <svg
              style={{
                position: 'absolute',
                left: 0,
                top: 0,
                width: '100%',
                height: '100%',
                overflow: 'visible',
              }}
            >
              <line
                x1={leftX}
                y1={y + 12}
                x2={rightX}
                y2={y + 12}
                stroke="rgba(255,255,255,0.4)"
                strokeWidth="2"
                strokeDasharray="4,4"
                style={{
                  filter: 'drop-shadow(0 0 3px rgba(255,255,255,0.5))',
                }}
              />
            </svg>

            {/* Backbone connections (vertical lines) */}
            {i < nucleotides - 1 && (
              <svg
                style={{
                  position: 'absolute',
                  left: 0,
                  top: 0,
                  width: '100%',
                  height: '100%',
                  overflow: 'visible',
                }}
              >
                {/* Left backbone */}
                <line
                  x1={leftX}
                  y1={y + 12}
                  x2={Math.cos(((i + 1) / nucleotides) * Math.PI * 4) * helixRadius + helixRadius * 1.5}
                  y2={(i + 1) * verticalSpacing + 12}
                  stroke="url(#backboneGradientLeft)"
                  strokeWidth="4"
                  strokeLinecap="round"
                  opacity={leftOpacity * 0.8}
                />
                {/* Right backbone */}
                <line
                  x1={rightX}
                  y1={y + 12}
                  x2={Math.cos(((i + 1) / nucleotides) * Math.PI * 4 + Math.PI) * helixRadius + helixRadius * 1.5}
                  y2={(i + 1) * verticalSpacing + 12}
                  stroke="url(#backboneGradientRight)"
                  strokeWidth="4"
                  strokeLinecap="round"
                  opacity={rightOpacity * 0.8}
                />
              </svg>
            )}
          </div>
        )
      })}

      {/* SVG Gradients */}
      <svg style={{ position: 'absolute', width: 0, height: 0 }}>
        <defs>
          <linearGradient id="backboneGradientLeft" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#06b6d4" />
            <stop offset="50%" stopColor="#8b5cf6" />
            <stop offset="100%" stopColor="#06b6d4" />
          </linearGradient>
          <linearGradient id="backboneGradientRight" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#f472b6" />
            <stop offset="50%" stopColor="#c084fc" />
            <stop offset="100%" stopColor="#f472b6" />
          </linearGradient>
        </defs>
      </svg>
    </div>
  )
}
