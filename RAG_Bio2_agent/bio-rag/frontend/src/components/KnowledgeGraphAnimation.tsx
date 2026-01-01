/**
 * Knowledge Graph Animation Component
 * 3D visualization of Neo4j knowledge graph with nodes and edges
 */

import { useRef, useState, useMemo, useEffect, useCallback } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { Text, OrbitControls, Line, Sphere, Html } from '@react-three/drei'
import * as THREE from 'three'
import { Play, Pause, Zap, Search, Loader2, Database, RefreshCw, FileText, User, Tag } from 'lucide-react'
import { graphApi, type GraphNode, type GraphEdge, type GraphStats } from '@/services/api'

// Node type colors
const NODE_COLORS: Record<string, string> = {
  paper: '#06b6d4',    // cyan for papers
  author: '#f59e0b',   // amber for authors
  keyword: '#10b981',  // emerald for keywords
}

// Edge type colors
const EDGE_COLORS: Record<string, string> = {
  cites: '#ef4444',       // red for citations
  authored: '#8b5cf6',    // purple for authorship
  has_keyword: '#22c55e', // green for keywords
}

// Node type icons
const NODE_ICONS: Record<string, string> = {
  paper: '📄',
  author: '👤',
  keyword: '🏷️',
}

// 3D Node with physics-like positioning
interface Node3D {
  id: string
  type: 'paper' | 'author' | 'keyword'
  label: string
  position: THREE.Vector3
  targetPosition: THREE.Vector3
  color: string
  data: GraphNode
}

interface Edge3D {
  source: string
  target: string
  type: string
  color: string
}

// Convert API nodes to 3D nodes with positions
function convertToNodes3D(nodes: GraphNode[]): Node3D[] {
  return nodes.map((node, idx) => {
    // Calculate position based on node type for better clustering
    const typeOffset = {
      paper: 0,
      author: Math.PI * 2 / 3,
      keyword: Math.PI * 4 / 3,
    }[node.type] || 0

    const angle = (idx / nodes.length) * Math.PI * 2 + typeOffset
    const radius = 3 + Math.random() * 2
    const height = (Math.random() - 0.5) * 3

    const targetX = Math.cos(angle) * radius
    const targetY = height
    const targetZ = Math.sin(angle) * radius

    return {
      id: node.id,
      type: node.type,
      label: node.label,
      position: new THREE.Vector3(
        (Math.random() - 0.5) * 15,
        (Math.random() - 0.5) * 15,
        (Math.random() - 0.5) * 15
      ),
      targetPosition: new THREE.Vector3(targetX, targetY, targetZ),
      color: NODE_COLORS[node.type] || '#ffffff',
      data: node,
    }
  })
}

// Convert API edges to 3D edges
function convertToEdges3D(edges: GraphEdge[]): Edge3D[] {
  return edges.map(edge => ({
    source: edge.source,
    target: edge.target,
    type: edge.type,
    color: EDGE_COLORS[edge.type] || '#666666',
  }))
}

// 3D Node Component
function GraphNode3D({
  node,
  isHovered,
  onHover,
  isPlaying,
}: {
  node: Node3D
  isHovered: boolean
  onHover: (id: string | null) => void
  isPlaying: boolean
}) {
  const meshRef = useRef<THREE.Mesh>(null)
  const textRef = useRef<THREE.Mesh>(null)

  const baseSize = node.type === 'paper' ? 0.25 : 0.18

  useFrame(() => {
    if (!meshRef.current || !isPlaying) return

    const current = meshRef.current.position
    current.lerp(node.targetPosition, 0.03)

    if (textRef.current) {
      textRef.current.position.copy(current)
      textRef.current.position.y += baseSize + 0.15
    }

    node.position.copy(current)
  })

  return (
    <group>
      <Sphere
        ref={meshRef}
        args={[isHovered ? baseSize * 1.5 : baseSize, 24, 24]}
        position={node.position}
        onPointerOver={() => onHover(node.id)}
        onPointerOut={() => onHover(null)}
      >
        <meshStandardMaterial
          color={node.color}
          emissive={node.color}
          emissiveIntensity={isHovered ? 0.8 : 0.4}
          transparent
          opacity={isHovered ? 1 : 0.85}
        />
      </Sphere>

      <Text
        ref={textRef}
        position={[node.position.x, node.position.y + baseSize + 0.15, node.position.z]}
        fontSize={isHovered ? 0.18 : 0.12}
        color="white"
        anchorX="center"
        anchorY="bottom"
        outlineWidth={0.02}
        outlineColor="#000000"
        maxWidth={2}
      >
        {node.label.length > 25 ? node.label.substring(0, 25) + '...' : node.label}
      </Text>

      {isHovered && (
        <Html position={[node.position.x, node.position.y - baseSize - 0.3, node.position.z]} center>
          <div className="px-3 py-2 bg-black/90 rounded-lg text-xs whitespace-nowrap border max-w-[200px]"
            style={{ borderColor: node.color }}>
            <div className="flex items-center gap-1 mb-1">
              <span>{NODE_ICONS[node.type]}</span>
              <span className="font-bold" style={{ color: node.color }}>
                {node.type.toUpperCase()}
              </span>
            </div>
            <div className="text-white/80 text-[10px] break-words">
              {node.data.title || node.data.name || node.data.term || node.label}
            </div>
            {node.data.pmid && (
              <div className="text-cyan-400 text-[10px] mt-1">PMID: {node.data.pmid}</div>
            )}
            {node.data.journal && (
              <div className="text-white/50 text-[10px]">{node.data.journal}</div>
            )}
          </div>
        </Html>
      )}
    </group>
  )
}

// Connection Lines
function EdgeLines({
  edges,
  nodes,
  hoveredNode,
}: {
  edges: Edge3D[]
  nodes: Node3D[]
  hoveredNode: string | null
}) {
  return (
    <>
      {edges.map((edge, i) => {
        const fromNode = nodes.find((n) => n.id === edge.source)
        const toNode = nodes.find((n) => n.id === edge.target)
        if (!fromNode || !toNode) return null

        const isHighlighted = hoveredNode === edge.source || hoveredNode === edge.target
        const opacity = isHighlighted ? 0.9 : 0.25
        const lineWidth = isHighlighted ? 2 : 0.8

        return (
          <group key={i}>
            <Line
              points={[fromNode.position, toNode.position]}
              color={isHighlighted ? edge.color : '#666666'}
              lineWidth={lineWidth}
              transparent
              opacity={opacity}
            />
            {isHighlighted && (
              <Html
                position={[
                  (fromNode.position.x + toNode.position.x) / 2,
                  (fromNode.position.y + toNode.position.y) / 2 + 0.15,
                  (fromNode.position.z + toNode.position.z) / 2,
                ]}
                center
              >
                <div
                  className="px-2 py-0.5 rounded text-[9px] text-white font-bold"
                  style={{ backgroundColor: edge.color + 'cc' }}
                >
                  {edge.type.replace('_', ' ').toUpperCase()}
                </div>
              </Html>
            )}
          </group>
        )
      })}
    </>
  )
}

// Grid and axes
function SceneHelpers() {
  return (
    <group>
      <Line points={[[-6, 0, 0], [6, 0, 0]]} color="#ff6b6b" lineWidth={1} transparent opacity={0.1} />
      <Line points={[[0, -6, 0], [0, 6, 0]]} color="#51cf66" lineWidth={1} transparent opacity={0.1} />
      <Line points={[[0, 0, -6], [0, 0, 6]]} color="#339af0" lineWidth={1} transparent opacity={0.1} />
      <gridHelper args={[12, 12, '#333333', '#222222']} />
    </group>
  )
}

// Main 3D Scene
function Scene({
  nodes,
  edges,
  hoveredNode,
  setHoveredNode,
  isPlaying,
  showEdges,
}: {
  nodes: Node3D[]
  edges: Edge3D[]
  hoveredNode: string | null
  setHoveredNode: (id: string | null) => void
  isPlaying: boolean
  showEdges: boolean
}) {
  return (
    <>
      <ambientLight intensity={0.6} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <pointLight position={[-10, -10, -10]} intensity={0.4} />

      <SceneHelpers />

      {showEdges && (
        <EdgeLines
          edges={edges}
          nodes={nodes}
          hoveredNode={hoveredNode}
        />
      )}

      {nodes.map((node) => (
        <GraphNode3D
          key={node.id}
          node={node}
          isHovered={hoveredNode === node.id}
          onHover={setHoveredNode}
          isPlaying={isPlaying}
        />
      ))}

      <OrbitControls
        enableDamping
        dampingFactor={0.05}
        rotateSpeed={0.5}
        minDistance={4}
        maxDistance={20}
      />
    </>
  )
}

// Stats display component
function StatsDisplay({ stats }: { stats: GraphStats | null }) {
  if (!stats || stats.status !== 'connected') {
    return (
      <div className="text-yellow-400 text-sm">
        GraphDB 연결 안됨
      </div>
    )
  }

  return (
    <div className="flex items-center gap-4 text-xs text-white/60">
      <div className="flex items-center gap-1">
        <FileText size={14} className="text-cyan-400" />
        <span>논문: {stats.papers}</span>
      </div>
      <div className="flex items-center gap-1">
        <User size={14} className="text-amber-400" />
        <span>저자: {stats.authors}</span>
      </div>
      <div className="flex items-center gap-1">
        <Tag size={14} className="text-emerald-400" />
        <span>키워드: {stats.keywords}</span>
      </div>
      <div className="text-white/40">
        인용: {stats.citations} | 관계: {stats.authorships + stats.keyword_links}
      </div>
    </div>
  )
}

// Main component
export default function KnowledgeGraphAnimation() {
  const [searchQuery, setSearchQuery] = useState('')
  const [isPlaying, setIsPlaying] = useState(true)
  const [showEdges, setShowEdges] = useState(true)
  const [hoveredNode, setHoveredNode] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Graph data state
  const [nodes, setNodes] = useState<Node3D[]>([])
  const [edges, setEdges] = useState<Edge3D[]>([])
  const [stats, setStats] = useState<GraphStats | null>(null)
  const [nodeFilter, setNodeFilter] = useState<'all' | 'paper' | 'author' | 'keyword'>('all')

  // Filtered nodes based on type
  const filteredNodes = useMemo(() => {
    if (nodeFilter === 'all') return nodes
    return nodes.filter(n => n.type === nodeFilter)
  }, [nodes, nodeFilter])

  // Filtered edges based on visible nodes
  const filteredEdges = useMemo(() => {
    const visibleNodeIds = new Set(filteredNodes.map(n => n.id))
    return edges.filter(e => visibleNodeIds.has(e.source) && visibleNodeIds.has(e.target))
  }, [edges, filteredNodes])

  // Fetch graph data
  const fetchGraphData = useCallback(async (query?: string) => {
    setIsLoading(true)
    setError(null)

    try {
      const [vizData, statsData] = await Promise.all([
        graphApi.getVisualization(query, 50),
        graphApi.getStats(),
      ])

      if (vizData.status === 'error') {
        setError(vizData.error || 'GraphDB 연결 오류')
        return
      }

      // Convert to 3D format
      const nodes3D = convertToNodes3D(vizData.nodes)
      const edges3D = convertToEdges3D(vizData.edges)

      setNodes(nodes3D)
      setEdges(edges3D)
      setStats(statsData)

      if (nodes3D.length === 0) {
        setError('그래프 데이터가 없습니다. GraphDB에 데이터를 먼저 인덱싱하세요.')
      }
    } catch (err) {
      console.error('Graph data fetch error:', err)
      setError('GraphDB에서 데이터를 가져올 수 없습니다.')
    } finally {
      setIsLoading(false)
    }
  }, [])

  // Initial load
  useEffect(() => {
    fetchGraphData()
  }, [fetchGraphData])

  // Handle search
  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault()
    await fetchGraphData(searchQuery.trim() || undefined)
  }

  // Randomize positions
  const handleRandomize = () => {
    setNodes(prev => prev.map(node => ({
      ...node,
      position: new THREE.Vector3(
        (Math.random() - 0.5) * 15,
        (Math.random() - 0.5) * 15,
        (Math.random() - 0.5) * 15
      ),
    })))
  }

  // Count nodes by type
  const nodeCounts = useMemo(() => ({
    paper: nodes.filter(n => n.type === 'paper').length,
    author: nodes.filter(n => n.type === 'author').length,
    keyword: nodes.filter(n => n.type === 'keyword').length,
  }), [nodes])

  return (
    <div className="glossy-panel p-6">
      <div className="flex items-center justify-between mb-4 flex-wrap gap-4">
        <div>
          <h2 className="text-xl font-semibold text-white flex items-center gap-2">
            <span className="text-2xl">🕸️</span>
            지식 그래프 (Knowledge Graph)
            <span className="px-2 py-0.5 text-xs bg-purple-500/20 text-purple-400 rounded-full border border-purple-400/30">
              Neo4j
            </span>
          </h2>
          <p className="text-white/60 text-sm mt-1">
            논문, 저자, 키워드 간의 관계 네트워크
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => fetchGraphData()}
            disabled={isLoading}
            className="p-2 rounded-lg bg-green-500/20 text-green-400 border border-green-400/30 hover:bg-green-500/30 transition-all disabled:opacity-50"
            title="새로고침"
          >
            <RefreshCw size={18} className={isLoading ? 'animate-spin' : ''} />
          </button>
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className={`p-2 rounded-lg transition-all ${
              isPlaying ? 'bg-orange-500/20 text-orange-400 border border-orange-400/30'
                : 'bg-green-500/20 text-green-400 border border-green-400/30'
            }`}
          >
            {isPlaying ? <Pause size={20} /> : <Play size={20} />}
          </button>
          <button
            onClick={handleRandomize}
            className="p-2 rounded-lg bg-purple-500/20 text-purple-400 border border-purple-400/30 hover:bg-purple-500/30 transition-all"
            title="위치 초기화"
          >
            <Zap size={20} />
          </button>
          <button
            onClick={() => setShowEdges(!showEdges)}
            className={`px-3 py-2 rounded-lg text-sm transition-all ${
              showEdges
                ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-400/30'
                : 'bg-white/10 text-white/70 border border-white/20'
            }`}
          >
            연결선
          </button>
        </div>
      </div>

      {/* Stats Display */}
      <div className="mb-4">
        <StatsDisplay stats={stats} />
      </div>

      {/* Error Message */}
      {error && (
        <div className="mb-4 p-3 rounded-lg bg-yellow-500/10 border border-yellow-500/30 flex items-center gap-2">
          <Database size={18} className="text-yellow-400" />
          <span className="text-yellow-400 text-sm">{error}</span>
        </div>
      )}

      {/* Node Type Filter */}
      <div className="flex items-center gap-2 mb-4 p-3 rounded-lg bg-white/5 border border-white/10 flex-wrap">
        <span className="text-sm text-white/70">노드 필터:</span>
        {(['all', 'paper', 'author', 'keyword'] as const).map((type) => {
          const count = type === 'all' ? nodes.length : nodeCounts[type]
          const color = type === 'all' ? '#ffffff' : NODE_COLORS[type]
          const icon = type === 'all' ? '🔵' : NODE_ICONS[type]

          return (
            <button
              key={type}
              onClick={() => setNodeFilter(type)}
              className={`px-3 py-1 rounded-lg text-xs font-medium transition-all flex items-center gap-1 ${
                nodeFilter === type
                  ? 'text-white'
                  : 'bg-white/5 text-white/40 border border-white/10'
              }`}
              style={{
                backgroundColor: nodeFilter === type ? color + '40' : undefined,
                borderColor: nodeFilter === type ? color : undefined,
              }}
            >
              <span>{icon}</span>
              <span>{type === 'all' ? '전체' : type === 'paper' ? '논문' : type === 'author' ? '저자' : '키워드'}</span>
              <span className="text-white/60">({count})</span>
            </button>
          )
        })}
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
              placeholder="논문 제목이나 키워드로 검색..."
              className="w-full pl-10 pr-4 py-2 rounded-lg bg-white/10 border border-white/20 text-white placeholder-white/40 focus:outline-none focus:border-purple-400/50"
              disabled={isLoading}
            />
          </div>
          <button
            type="submit"
            disabled={isLoading}
            className="px-4 py-2 rounded-lg bg-purple-500/20 text-purple-400 border border-purple-400/30 hover:bg-purple-500/30 transition-all disabled:opacity-50 flex items-center gap-2"
          >
            {isLoading ? (
              <>
                <Loader2 size={16} className="animate-spin" />
                검색중...
              </>
            ) : (
              '검색'
            )}
          </button>
        </div>
      </form>

      {/* 3D Canvas */}
      <div className="relative w-full h-[520px] rounded-xl overflow-hidden bg-slate-900/80 border border-white/10">
        {/* Loading Overlay */}
        {isLoading && (
          <div className="absolute inset-0 z-10 flex items-center justify-center bg-slate-900/90 backdrop-blur-sm">
            <div className="text-center">
              <Loader2 size={48} className="animate-spin text-purple-400 mx-auto mb-4" />
              <p className="text-white/80 font-medium">지식 그래프 로딩 중...</p>
              <p className="text-white/50 text-sm mt-1">Neo4j에서 데이터 가져오는 중</p>
            </div>
          </div>
        )}

        <Canvas camera={{ position: [8, 5, 8], fov: 50 }}>
          <Scene
            nodes={filteredNodes}
            edges={filteredEdges}
            hoveredNode={hoveredNode}
            setHoveredNode={setHoveredNode}
            isPlaying={isPlaying}
            showEdges={showEdges}
          />
        </Canvas>

        {/* Legend */}
        <div className="absolute bottom-4 left-4 p-3 rounded-lg bg-black/80 backdrop-blur-sm border border-white/10">
          <div className="text-xs text-white/60 mb-2">범례</div>
          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: NODE_COLORS.paper }} />
              <span className="text-xs text-white/70">논문 (Paper)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: NODE_COLORS.author }} />
              <span className="text-xs text-white/70">저자 (Author)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: NODE_COLORS.keyword }} />
              <span className="text-xs text-white/70">키워드 (Keyword)</span>
            </div>
          </div>
          <div className="mt-2 pt-2 border-t border-white/10 flex flex-col gap-1">
            <div className="flex items-center gap-2">
              <div className="w-4 h-0.5" style={{ backgroundColor: EDGE_COLORS.cites }} />
              <span className="text-xs text-white/50">인용 (CITES)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-0.5" style={{ backgroundColor: EDGE_COLORS.authored }} />
              <span className="text-xs text-white/50">저술 (AUTHORED)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-0.5" style={{ backgroundColor: EDGE_COLORS.has_keyword }} />
              <span className="text-xs text-white/50">키워드 (HAS_KEYWORD)</span>
            </div>
          </div>
        </div>

        {/* Hovered Node Info */}
        {hoveredNode && (
          <div className="absolute top-4 right-4 p-3 rounded-lg bg-black/85 backdrop-blur-sm border border-white/20">
            <div className="text-white/90 font-medium text-sm">
              {filteredNodes.find((n) => n.id === hoveredNode)?.label}
            </div>
            <div className="text-white/50 text-xs mt-1">
              유형: {filteredNodes.find((n) => n.id === hoveredNode)?.type}
            </div>
          </div>
        )}
      </div>

      {/* Info Text */}
      <div className="mt-4 text-xs text-white/40 text-center">
        마우스로 드래그하여 회전 | 스크롤로 확대/축소 | 노드를 호버하여 상세 정보 확인
      </div>
    </div>
  )
}
