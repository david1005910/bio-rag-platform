import { useRef, useState, useMemo, useEffect } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { Text, OrbitControls, Line, Sphere } from '@react-three/drei'
import * as THREE from 'three'
import { Play, Pause, Zap, Search } from 'lucide-react'

// Related words database - maps search terms to related concepts
const RELATED_WORDS_DB: Record<string, { words: string[]; color: string }[]> = {
  cancer: [
    { words: ['cancer', 'tumor', 'malignant', 'carcinoma', 'oncology'], color: '#f43f5e' },
    { words: ['metastasis', 'invasion', 'progression', 'staging'], color: '#fb923c' },
    { words: ['chemotherapy', 'radiation', 'surgery', 'treatment'], color: '#22c55e' },
    { words: ['biomarker', 'diagnosis', 'prognosis', 'screening'], color: '#06b6d4' },
  ],
  immunotherapy: [
    { words: ['immunotherapy', 'immune', 'checkpoint', 'PD-1', 'PD-L1'], color: '#8b5cf6' },
    { words: ['CAR-T', 'T-cell', 'NK cell', 'lymphocyte'], color: '#06b6d4' },
    { words: ['antibody', 'antigen', 'cytokine', 'interferon'], color: '#22c55e' },
    { words: ['response', 'resistance', 'efficacy', 'toxicity'], color: '#f59e0b' },
  ],
  crispr: [
    { words: ['CRISPR', 'Cas9', 'Cas12', 'guide RNA', 'PAM'], color: '#8b5cf6' },
    { words: ['gene editing', 'knockout', 'knockin', 'HDR'], color: '#06b6d4' },
    { words: ['off-target', 'specificity', 'efficiency', 'delivery'], color: '#f59e0b' },
    { words: ['therapy', 'correction', 'modification', 'engineering'], color: '#22c55e' },
  ],
  'deep learning': [
    { words: ['deep learning', 'neural network', 'CNN', 'transformer'], color: '#8b5cf6' },
    { words: ['training', 'inference', 'optimization', 'gradient'], color: '#06b6d4' },
    { words: ['AlphaFold', 'protein', 'structure', 'prediction'], color: '#22c55e' },
    { words: ['accuracy', 'loss', 'validation', 'benchmark'], color: '#f59e0b' },
  ],
  protein: [
    { words: ['protein', 'amino acid', 'peptide', 'polypeptide'], color: '#f59e0b' },
    { words: ['folding', 'structure', '3D', 'conformation'], color: '#8b5cf6' },
    { words: ['enzyme', 'receptor', 'antibody', 'ligand'], color: '#06b6d4' },
    { words: ['binding', 'interaction', 'affinity', 'specificity'], color: '#22c55e' },
  ],
  rna: [
    { words: ['RNA', 'mRNA', 'siRNA', 'miRNA', 'lncRNA'], color: '#f43f5e' },
    { words: ['transcription', 'splicing', 'translation', 'degradation'], color: '#8b5cf6' },
    { words: ['vaccine', 'therapeutic', 'delivery', 'LNP'], color: '#22c55e' },
    { words: ['expression', 'regulation', 'silencing', 'knockdown'], color: '#06b6d4' },
  ],
  default: [
    { words: ['research', 'study', 'analysis', 'method', 'result'], color: '#06b6d4' },
    { words: ['data', 'sample', 'experiment', 'protocol'], color: '#8b5cf6' },
    { words: ['finding', 'discovery', 'insight', 'conclusion'], color: '#22c55e' },
    { words: ['publication', 'journal', 'review', 'citation'], color: '#f59e0b' },
  ],
}

interface WordNode {
  id: string
  text: string
  position: THREE.Vector3
  targetPosition: THREE.Vector3
  color: string
  groupIndex: number
}

interface Connection {
  from: string
  to: string
  strength: number
}

// Generate connections between words in same/related groups
function generateConnections(words: WordNode[]): Connection[] {
  const connections: Connection[] = []

  words.forEach((word1, i) => {
    words.forEach((word2, j) => {
      if (i >= j) return

      // Same group = high similarity
      if (word1.groupIndex === word2.groupIndex) {
        connections.push({
          from: word1.id,
          to: word2.id,
          strength: 0.7 + Math.random() * 0.3,
        })
      }
      // Adjacent groups = medium similarity
      else if (Math.abs(word1.groupIndex - word2.groupIndex) === 1) {
        if (Math.random() > 0.6) {
          connections.push({
            from: word1.id,
            to: word2.id,
            strength: 0.3 + Math.random() * 0.3,
          })
        }
      }
    })
  })

  return connections
}

// 3D Word Node Component
function WordNode3D({
  node,
  isHovered,
  onHover,
  isPlaying,
}: {
  node: WordNode
  isHovered: boolean
  onHover: (id: string | null) => void
  isPlaying: boolean
}) {
  const meshRef = useRef<THREE.Mesh>(null)
  const textRef = useRef<THREE.Mesh>(null)

  useFrame(() => {
    if (!meshRef.current || !isPlaying) return

    // Animate towards target position
    const current = meshRef.current.position
    current.lerp(node.targetPosition, 0.02)

    // Sync text position
    if (textRef.current) {
      textRef.current.position.copy(current)
      textRef.current.position.y += 0.4
    }

    // Update node position for line drawing
    node.position.copy(current)
  })

  return (
    <group>
      {/* Sphere */}
      <Sphere
        ref={meshRef}
        args={[isHovered ? 0.35 : 0.25, 16, 16]}
        position={node.position}
        onPointerOver={() => onHover(node.id)}
        onPointerOut={() => onHover(null)}
      >
        <meshStandardMaterial
          color={node.color}
          emissive={node.color}
          emissiveIntensity={isHovered ? 0.5 : 0.2}
          transparent
          opacity={isHovered ? 1 : 0.8}
        />
      </Sphere>

      {/* Text Label */}
      <Text
        ref={textRef}
        position={[node.position.x, node.position.y + 0.4, node.position.z]}
        fontSize={isHovered ? 0.25 : 0.18}
        color="white"
        anchorX="center"
        anchorY="bottom"
        outlineWidth={0.02}
        outlineColor="#000000"
      >
        {node.text}
      </Text>
    </group>
  )
}

// Connection Lines Component
function ConnectionLines({
  connections,
  nodes,
  hoveredWord,
}: {
  connections: Connection[]
  nodes: WordNode[]
  hoveredWord: string | null
}) {
  return (
    <>
      {connections.map((conn, i) => {
        const fromNode = nodes.find((n) => n.id === conn.from)
        const toNode = nodes.find((n) => n.id === conn.to)
        if (!fromNode || !toNode) return null

        const isHighlighted = hoveredWord === conn.from || hoveredWord === conn.to
        const opacity = isHighlighted ? conn.strength : conn.strength * 0.3

        return (
          <Line
            key={i}
            points={[fromNode.position, toNode.position]}
            color={isHighlighted ? '#06b6d4' : '#ffffff'}
            lineWidth={isHighlighted ? 2 : 1}
            transparent
            opacity={opacity}
          />
        )
      })}
    </>
  )
}

// Axes Helper
function Axes() {
  return (
    <group>
      {/* X Axis */}
      <Line points={[[-5, 0, 0], [5, 0, 0]]} color="#ff6b6b" lineWidth={1} transparent opacity={0.3} />
      <Text position={[5.5, 0, 0]} fontSize={0.2} color="#ff6b6b">X</Text>

      {/* Y Axis */}
      <Line points={[[0, -5, 0], [0, 5, 0]]} color="#51cf66" lineWidth={1} transparent opacity={0.3} />
      <Text position={[0, 5.5, 0]} fontSize={0.2} color="#51cf66">Y</Text>

      {/* Z Axis */}
      <Line points={[[0, 0, -5], [0, 0, 5]]} color="#339af0" lineWidth={1} transparent opacity={0.3} />
      <Text position={[0, 0, 5.5]} fontSize={0.2} color="#339af0">Z</Text>

      {/* Grid */}
      <gridHelper args={[10, 10, '#333333', '#222222']} rotation={[0, 0, 0]} />
    </group>
  )
}

// Main Scene Component
function Scene({
  nodes,
  connections,
  hoveredWord,
  setHoveredWord,
  isPlaying,
  showConnections,
}: {
  nodes: WordNode[]
  connections: Connection[]
  hoveredWord: string | null
  setHoveredWord: (id: string | null) => void
  isPlaying: boolean
  showConnections: boolean
}) {
  return (
    <>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <pointLight position={[-10, -10, -10]} intensity={0.5} />

      <Axes />

      {showConnections && (
        <ConnectionLines connections={connections} nodes={nodes} hoveredWord={hoveredWord} />
      )}

      {nodes.map((node) => (
        <WordNode3D
          key={node.id}
          node={node}
          isHovered={hoveredWord === node.id}
          onHover={setHoveredWord}
          isPlaying={isPlaying}
        />
      ))}

      <OrbitControls
        enableDamping
        dampingFactor={0.05}
        rotateSpeed={0.5}
        minDistance={3}
        maxDistance={20}
      />
    </>
  )
}

export default function VectorSpaceAnimation() {
  const [searchQuery, setSearchQuery] = useState('')
  const [activeQuery, setActiveQuery] = useState('cancer')
  const [isPlaying, setIsPlaying] = useState(true)
  const [showConnections, setShowConnections] = useState(true)
  const [hoveredWord, setHoveredWord] = useState<string | null>(null)

  // Generate nodes based on search query
  const { nodes, connections } = useMemo(() => {
    const queryLower = activeQuery.toLowerCase()
    const wordGroups = RELATED_WORDS_DB[queryLower] || RELATED_WORDS_DB.default

    const generatedNodes: WordNode[] = []

    wordGroups.forEach((group, groupIndex) => {
      // Calculate cluster center in 3D space
      const angle = (groupIndex / wordGroups.length) * Math.PI * 2
      const radius = 2.5
      const clusterCenter = new THREE.Vector3(
        Math.cos(angle) * radius,
        (groupIndex - wordGroups.length / 2) * 0.8,
        Math.sin(angle) * radius
      )

      group.words.forEach((word, wordIndex) => {
        // Random initial position
        const startPos = new THREE.Vector3(
          (Math.random() - 0.5) * 10,
          (Math.random() - 0.5) * 10,
          (Math.random() - 0.5) * 10
        )

        // Target position near cluster center
        const spreadAngle = (wordIndex / group.words.length) * Math.PI * 2
        const spreadRadius = 0.8 + Math.random() * 0.4
        const targetPos = new THREE.Vector3(
          clusterCenter.x + Math.cos(spreadAngle) * spreadRadius,
          clusterCenter.y + (Math.random() - 0.5) * 0.5,
          clusterCenter.z + Math.sin(spreadAngle) * spreadRadius
        )

        generatedNodes.push({
          id: word,
          text: word,
          position: startPos,
          targetPosition: targetPos,
          color: group.color,
          groupIndex,
        })
      })
    })

    const generatedConnections = generateConnections(generatedNodes)

    return { nodes: generatedNodes, connections: generatedConnections }
  }, [activeQuery])

  // Reset node positions when query changes
  useEffect(() => {
    nodes.forEach((node) => {
      node.position.set(
        (Math.random() - 0.5) * 10,
        (Math.random() - 0.5) * 10,
        (Math.random() - 0.5) * 10
      )
    })
  }, [activeQuery, nodes])

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (searchQuery.trim()) {
      setActiveQuery(searchQuery.trim())
    }
  }

  const handleRandomize = () => {
    nodes.forEach((node) => {
      node.position.set(
        (Math.random() - 0.5) * 10,
        (Math.random() - 0.5) * 10,
        (Math.random() - 0.5) * 10
      )
    })
  }

  const presetQueries = ['cancer', 'immunotherapy', 'crispr', 'deep learning', 'protein', 'rna']

  return (
    <div className="glossy-panel p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-4 flex-wrap gap-4">
        <div>
          <h2 className="text-xl font-semibold text-white flex items-center gap-2">
            <span className="text-2xl">🧬</span>
            3D 벡터 스페이스
          </h2>
          <p className="text-white/60 text-sm mt-1">
            "{activeQuery}" 관련 단어들의 임베딩 시각화
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className={`p-2 rounded-lg transition-all ${
              isPlaying
                ? 'bg-orange-500/20 text-orange-400 border border-orange-400/30'
                : 'bg-green-500/20 text-green-400 border border-green-400/30'
            }`}
          >
            {isPlaying ? <Pause size={20} /> : <Play size={20} />}
          </button>
          <button
            onClick={handleRandomize}
            className="p-2 rounded-lg bg-purple-500/20 text-purple-400 border border-purple-400/30 hover:bg-purple-500/30 transition-all"
            title="재배치"
          >
            <Zap size={20} />
          </button>
          <button
            onClick={() => setShowConnections(!showConnections)}
            className={`px-3 py-2 rounded-lg text-sm transition-all ${
              showConnections
                ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-400/30'
                : 'bg-white/10 text-white/70 border border-white/20'
            }`}
          >
            연결선 {showConnections ? 'ON' : 'OFF'}
          </button>
        </div>
      </div>

      {/* Search Bar */}
      <form onSubmit={handleSearch} className="mb-4">
        <div className="flex gap-2">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-white/50" size={18} />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="검색어 입력 (예: cancer, protein, RNA...)"
              className="w-full pl-10 pr-4 py-2 rounded-lg bg-white/10 border border-white/20 text-white placeholder-white/40 focus:outline-none focus:border-cyan-400/50"
            />
          </div>
          <button
            type="submit"
            className="px-4 py-2 rounded-lg bg-cyan-500/20 text-cyan-400 border border-cyan-400/30 hover:bg-cyan-500/30 transition-all"
          >
            검색
          </button>
        </div>
      </form>

      {/* Preset Query Buttons */}
      <div className="flex flex-wrap gap-2 mb-4">
        {presetQueries.map((query) => (
          <button
            key={query}
            onClick={() => setActiveQuery(query)}
            className={`px-3 py-1.5 rounded-full text-xs font-medium transition-all ${
              activeQuery === query
                ? 'bg-cyan-500/30 text-cyan-400 border border-cyan-400/50'
                : 'bg-white/5 text-white/60 border border-white/10 hover:bg-white/10'
            }`}
          >
            {query}
          </button>
        ))}
      </div>

      {/* 3D Canvas */}
      <div className="relative w-full h-[500px] rounded-xl overflow-hidden bg-slate-900/80 border border-white/10">
        <Canvas camera={{ position: [6, 4, 6], fov: 50 }}>
          <Scene
            nodes={nodes}
            connections={connections}
            hoveredWord={hoveredWord}
            setHoveredWord={setHoveredWord}
            isPlaying={isPlaying}
            showConnections={showConnections}
          />
        </Canvas>

        {/* Hovered Word Info */}
        {hoveredWord && (
          <div className="absolute top-4 left-4 p-3 rounded-lg bg-black/70 backdrop-blur-sm border border-cyan-400/30">
            <div className="text-cyan-400 font-medium">{hoveredWord}</div>
            <div className="text-xs text-white/60 mt-1">
              연결: {connections.filter((c) => c.from === hoveredWord || c.to === hoveredWord).length}개
            </div>
          </div>
        )}

        {/* Controls Hint */}
        <div className="absolute bottom-4 right-4 p-2 rounded-lg bg-black/50 text-xs text-white/50">
          마우스 드래그: 회전 | 스크롤: 확대/축소
        </div>
      </div>

      {/* Info Panel */}
      <div className="mt-4 p-4 rounded-xl bg-white/5 border border-white/10">
        <h3 className="text-sm font-medium text-white mb-2">3D 벡터 스페이스란?</h3>
        <p className="text-xs text-white/70 leading-relaxed">
          검색어와 관련된 단어들이 의미적 유사도에 따라 3차원 공간에 배치됩니다.
          가까운 단어일수록 의미가 유사하며, 같은 색상의 단어들은 동일한 개념 클러스터에 속합니다.
          마우스로 공간을 회전하고 확대하여 단어 간의 관계를 탐색해보세요.
        </p>
      </div>
    </div>
  )
}
