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
  const nucleotides = 24 // Number of nucleotide pairs (increased for smoother helix)
  const helixHeight = 650
  const helixRadius = 50 // Radius of the circular helix (constant)
  const verticalSpacing = helixHeight / nucleotides
  const containerWidth = helixRadius * 2 + 40 // Container width to fit the circular helix
  const centerX = containerWidth / 2 // Center of the circular helix

  // Base pair colors - vibrant for visibility
  const basePairs = [
    { left: '#ef4444', right: '#22c55e', label: 'A-T' }, // Adenine-Thymine (Red-Green)
    { left: '#3b82f6', right: '#eab308', label: 'G-C' }, // Guanine-Cytosine (Blue-Yellow)
  ]

  return (
    <div
      className="dna-3d-helix"
      style={{
        width: `${containerWidth}px`,
        height: `${helixHeight}px`,
        position: 'relative',
        transformStyle: 'preserve-3d',
        animation: `dna-spin 12s linear ${delay}s infinite`,
      }}
    >
      {Array.from({ length: nucleotides }).map((_, i) => {
        // Circular helix: both strands rotate around the central axis
        // maintaining constant radius (true cylindrical helix)
        const angle = (i / nucleotides) * Math.PI * 3 // 1.5 full rotations for smoother look
        const y = i * verticalSpacing
        const pair = basePairs[i % 2]

        // Calculate positions on a circle (constant radius from center)
        // Left strand: angle
        // Right strand: angle + PI (opposite side of the circle)
        const leftX = centerX + Math.cos(angle) * helixRadius
        const leftZ = Math.sin(angle) * helixRadius
        const rightX = centerX + Math.cos(angle + Math.PI) * helixRadius
        const rightZ = Math.sin(angle + Math.PI) * helixRadius

        // Next position for backbone connection
        const nextAngle = ((i + 1) / nucleotides) * Math.PI * 3
        const nextLeftX = centerX + Math.cos(nextAngle) * helixRadius
        const nextRightX = centerX + Math.cos(nextAngle + Math.PI) * helixRadius

        // Opacity and scale based on Z position (depth effect)
        // Front (positive Z) = brighter, larger
        // Back (negative Z) = dimmer, smaller
        const leftDepthFactor = (leftZ + helixRadius) / (helixRadius * 2) // 0 to 1
        const rightDepthFactor = (rightZ + helixRadius) / (helixRadius * 2) // 0 to 1

        const leftOpacity = 0.4 + leftDepthFactor * 0.6
        const rightOpacity = 0.4 + rightDepthFactor * 0.6

        const leftScale = 0.7 + leftDepthFactor * 0.5
        const rightScale = 0.7 + rightDepthFactor * 0.5

        const sphereSize = 20

        return (
          <div key={i} style={{ position: 'absolute', width: '100%', height: '100%' }}>
            {/* Left nucleotide (backbone sphere) */}
            <div
              className="nucleotide-sphere"
              style={{
                position: 'absolute',
                left: `${leftX - (sphereSize * leftScale) / 2}px`,
                top: `${y}px`,
                width: `${sphereSize * leftScale}px`,
                height: `${sphereSize * leftScale}px`,
                borderRadius: '50%',
                background: `radial-gradient(circle at 35% 35%, white, ${pair.left} 40%, ${pair.left}88 70%, ${pair.left}44)`,
                boxShadow: `0 0 ${15 * leftScale}px ${pair.left}90, inset 0 -3px 10px rgba(0,0,0,0.4)`,
                opacity: leftOpacity,
                zIndex: Math.round(leftZ + helixRadius),
                animation: `nucleotide-pulse 2.5s ease-in-out ${delay + i * 0.08}s infinite`,
              }}
            />

            {/* Right nucleotide (backbone sphere) */}
            <div
              className="nucleotide-sphere"
              style={{
                position: 'absolute',
                left: `${rightX - (sphereSize * rightScale) / 2}px`,
                top: `${y}px`,
                width: `${sphereSize * rightScale}px`,
                height: `${sphereSize * rightScale}px`,
                borderRadius: '50%',
                background: `radial-gradient(circle at 35% 35%, white, ${pair.right} 40%, ${pair.right}88 70%, ${pair.right}44)`,
                boxShadow: `0 0 ${15 * rightScale}px ${pair.right}90, inset 0 -3px 10px rgba(0,0,0,0.4)`,
                opacity: rightOpacity,
                zIndex: Math.round(rightZ + helixRadius),
                animation: `nucleotide-pulse 2.5s ease-in-out ${delay + i * 0.08 + 0.4}s infinite`,
              }}
            />

            {/* Hydrogen bond (connecting line between base pairs) */}
            <svg
              style={{
                position: 'absolute',
                left: 0,
                top: 0,
                width: '100%',
                height: '100%',
                overflow: 'visible',
                zIndex: Math.round((leftZ + rightZ) / 2 + helixRadius) - 1,
              }}
            >
              <line
                x1={leftX}
                y1={y + (sphereSize * leftScale) / 2}
                x2={rightX}
                y2={y + (sphereSize * rightScale) / 2}
                stroke="rgba(255,255,255,0.5)"
                strokeWidth="2"
                strokeDasharray="6,4"
                style={{
                  filter: 'drop-shadow(0 0 4px rgba(255,255,255,0.6))',
                }}
              />
            </svg>

            {/* Backbone connections (curved lines connecting nucleotides) */}
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
                {/* Left backbone - gradient tube effect */}
                <line
                  x1={leftX}
                  y1={y + (sphereSize * leftScale) / 2}
                  x2={nextLeftX}
                  y2={(i + 1) * verticalSpacing + (sphereSize * leftScale) / 2}
                  stroke="url(#backboneGradientLeft)"
                  strokeWidth={4 * leftScale}
                  strokeLinecap="round"
                  opacity={leftOpacity * 0.9}
                  style={{ zIndex: Math.round(leftZ + helixRadius) - 2 }}
                />
                {/* Right backbone - gradient tube effect */}
                <line
                  x1={rightX}
                  y1={y + (sphereSize * rightScale) / 2}
                  x2={nextRightX}
                  y2={(i + 1) * verticalSpacing + (sphereSize * rightScale) / 2}
                  stroke="url(#backboneGradientRight)"
                  strokeWidth={4 * rightScale}
                  strokeLinecap="round"
                  opacity={rightOpacity * 0.9}
                  style={{ zIndex: Math.round(rightZ + helixRadius) - 2 }}
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
