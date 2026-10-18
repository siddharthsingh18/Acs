import L from 'leaflet'

if (typeof window !== 'undefined') {
  ;(window as any).L = L
}

if (typeof globalThis !== 'undefined') {
  ;(globalThis as any).L = L
}

export default L
export { L }
