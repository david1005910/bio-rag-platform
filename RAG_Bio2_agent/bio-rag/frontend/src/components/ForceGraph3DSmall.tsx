import { useEffect, useRef, useState } from 'react'
import ForceGraph3D from '3d-force-graph'
import { graphApi } from '@/services/api'
import { Loader2, Maximize2, Minimize2, Square } from 'lucide-react'

// Size presets
type GraphSize = 'small' | 'medium' | 'large' | 'fullscreen'

const SIZE_CONFIG: Record<GraphSize, { height: string; label: string }> = {
  small: { height: '250px', label: 'S' },
  medium: { height: '400px', label: 'M' },
  large: { height: '600px', label: 'L' },
  fullscreen: { height: '85vh', label: 'Full' },
}

interface GraphNode {
  id: string
  name: string
  type: 'paper' | 'author' | 'keyword'
  val?: number
  color?: string
  x?: number
  y?: number
  z?: number
}

interface GraphLink {
  source: string
  target: string
  type: 'cites' | 'authored' | 'has_keyword'
  color?: string
}

interface GraphData {
  nodes: GraphNode[]
  links: GraphLink[]
}

// Node colors by type
const NODE_COLORS: Record<string, string> = {
  paper: '#06b6d4',    // cyan
  author: '#f59e0b',   // amber
  keyword: '#10b981',  // emerald
}

// Link colors by type
const LINK_COLORS: Record<string, string> = {
  cites: '#ef4444',       // red
  authored: '#a855f7',    // purple
  has_keyword: '#22c55e', // green
}

export default function ForceGraph3DSmall() {
  const containerRef = useRef<HTMLDivElement>(null)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const graphRef = useRef<any>(null)
  const [graphData, setGraphData] = useState<GraphData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null)
  const [graphSize, setGraphSize] = useState<GraphSize>('medium')

  // Fetch graph data
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        const data = await graphApi.getVisualization(undefined, 80)

        // Transform nodes
        const nodes: GraphNode[] = data.nodes.map((node) => ({
          id: node.id,
          name: node.label,
          type: node.type as 'paper' | 'author' | 'keyword',
          val: node.type === 'paper' ? 6 : node.type === 'author' ? 4 : 2,
          color: NODE_COLORS[node.type] || '#888888',
        }))

        // Transform links
        const links: GraphLink[] = data.edges.map((edge) => ({
          source: edge.source,
          target: edge.target,
          type: edge.type as 'cites' | 'authored' | 'has_keyword',
          color: LINK_COLORS[edge.type] || '#444444',
        }))

        setGraphData({ nodes, links })
        setError(null)
      } catch (err) {
        console.error('Failed to fetch graph data:', err)
        setError('그래프 데이터를 불러오는데 실패했습니다')
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [])

  // Initialize force graph
  useEffect(() => {
    if (!containerRef.current || !graphData) return

    const container = containerRef.current
    const { width, height } = container.getBoundingClientRect()

    // Create 3D force graph instance
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const Graph = new ForceGraph3D(container) as any
    Graph
      .graphData(graphData)
      .width(width)
      .height(height)
      .backgroundColor('rgba(15, 23, 42, 0.95)') // Dark background
      .nodeRelSize(3)
      .nodeVal((node: GraphNode) => node.val || 3)
      .nodeColor((node: GraphNode) => node.color || '#888888')
      .nodeLabel((node: GraphNode) => node.name)
      .nodeOpacity(0.9)
      .linkColor((link: GraphLink) => link.color || '#444444')
      .linkWidth(0.8)
      .linkOpacity(0.6)
      .linkDirectionalParticles(2)
      .linkDirectionalParticleWidth(1.5)
      .linkDirectionalParticleSpeed(0.006)
      .cooldownTicks(100)
      .onNodeHover((node: GraphNode | null) => {
        setHoveredNode(node)
        container.style.cursor = node ? 'pointer' : 'default'
      })
      .onNodeClick((node: GraphNode) => {
        // Focus on node with animation
        const distance = 80
        const distRatio = 1 + distance / Math.hypot(node.x || 0, node.y || 0, node.z || 0)
        const lookAtCoords = { x: node.x || 0, y: node.y || 0, z: node.z || 0 }
        Graph.cameraPosition(
          {
            x: (node.x || 0) * distRatio,
            y: (node.y || 0) * distRatio,
            z: (node.z || 0) * distRatio
          },
          lookAtCoords,
          1500
        )
      })

    // Set initial camera position
    Graph.cameraPosition({ x: 0, y: 0, z: 200 })

    // Enable auto-rotation
    let angle = 0
    const rotationInterval = setInterval(() => {
      angle += 0.002
      Graph.cameraPosition({
        x: 150 * Math.sin(angle),
        y: 50,
        z: 150 * Math.cos(angle)
      })
    }, 30)

    graphRef.current = Graph

    // Handle resize
    const resizeObserver = new ResizeObserver(() => {
      const { width, height } = container.getBoundingClientRect()
      Graph.width(width).height(height)
    })
    resizeObserver.observe(container)

    return () => {
      clearInterval(rotationInterval)
      resizeObserver.disconnect()
      if (graphRef.current) {
        graphRef.current._destructor()
        graphRef.current = null
      }
    }
  }, [graphData])

  if (loading) {
    return (
      <div
        className="glossy-panel p-6 flex items-center justify-center bg-slate-900/50"
        style={{ height: SIZE_CONFIG[graphSize].height }}
      >
        <div className="flex flex-col items-center gap-3">
          <Loader2 className="animate-spin text-cyan-500" size={32} />
          <span className="text-slate-300">3D 그래프 로딩 중...</span>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div
        className="glossy-panel p-6 flex items-center justify-center bg-slate-900/50"
        style={{ height: SIZE_CONFIG[graphSize].height }}
      >
        <span className="text-red-400">{error}</span>
      </div>
    )
  }

  return (
    <div className="glossy-panel p-4 relative overflow-hidden bg-slate-900/30">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-slate-800">3D 지식 그래프 (인터랙티브)</h3>
        <div className="flex items-center gap-4">
          {/* Size Controls */}
          <div className="flex items-center gap-1 bg-slate-100 rounded-lg p-1">
            <button
              onClick={() => setGraphSize('small')}
              className={`px-2 py-1 rounded text-xs font-medium transition-all ${
                graphSize === 'small'
                  ? 'bg-white text-slate-800 shadow-sm'
                  : 'text-slate-500 hover:text-slate-700'
              }`}
              title="작게"
            >
              <Minimize2 size={14} />
            </button>
            <button
              onClick={() => setGraphSize('medium')}
              className={`px-2 py-1 rounded text-xs font-medium transition-all ${
                graphSize === 'medium'
                  ? 'bg-white text-slate-800 shadow-sm'
                  : 'text-slate-500 hover:text-slate-700'
              }`}
              title="보통"
            >
              <Square size={14} />
            </button>
            <button
              onClick={() => setGraphSize('large')}
              className={`px-2 py-1 rounded text-xs font-medium transition-all ${
                graphSize === 'large'
                  ? 'bg-white text-slate-800 shadow-sm'
                  : 'text-slate-500 hover:text-slate-700'
              }`}
              title="크게"
            >
              <Maximize2 size={14} />
            </button>
            <button
              onClick={() => setGraphSize('fullscreen')}
              className={`px-2 py-1 rounded text-xs font-medium transition-all ${
                graphSize === 'fullscreen'
                  ? 'bg-cyan-500 text-white shadow-sm'
                  : 'text-slate-500 hover:text-slate-700'
              }`}
              title="전체화면"
            >
              Full
            </button>
          </div>

          {/* Legend */}
          <div className="flex items-center gap-3 text-sm">
            <div className="flex items-center gap-1">
              <div className="w-2.5 h-2.5 rounded-full bg-cyan-500" />
              <span className="text-slate-600 text-xs">논문</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-2.5 h-2.5 rounded-full bg-amber-500" />
              <span className="text-slate-600 text-xs">저자</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
              <span className="text-slate-600 text-xs">키워드</span>
            </div>
          </div>
        </div>
      </div>

      <div
        ref={containerRef}
        className="w-full rounded-lg overflow-hidden transition-all duration-300"
        style={{ height: SIZE_CONFIG[graphSize].height }}
      />

      {/* Hovered node info */}
      {hoveredNode && (
        <div className="absolute bottom-8 left-8 bg-slate-800/95 rounded-lg px-4 py-3 shadow-xl border border-slate-600 max-w-[280px]">
          <div className="text-sm font-medium text-white truncate">
            {hoveredNode.name}
          </div>
          <div className="text-xs text-slate-400 mt-1">
            {hoveredNode.type === 'paper' ? '논문' :
             hoveredNode.type === 'author' ? '저자' : '키워드'}
          </div>
        </div>
      )}

      {/* Graph stats */}
      {graphData && (
        <div className="mt-3 flex items-center justify-center gap-6 text-sm text-slate-600">
          <span>노드: {graphData.nodes.length}개</span>
          <span>엣지: {graphData.links.length}개</span>
          <span className="text-xs text-slate-400">(마우스 드래그로 회전, 클릭으로 포커스)</span>
        </div>
      )}
    </div>
  )
}
