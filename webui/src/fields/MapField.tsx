import { useEffect, useMemo, useRef, useState } from 'react'
import { FieldProps } from '@rjsf/utils'
import { Alert, Box, Button, Dialog, DialogActions, DialogContent, DialogTitle, FormControlLabel, Stack, Switch, TextField, ToggleButton, ToggleButtonGroup, Typography } from '@mui/material'
import MapIcon from '@mui/icons-material/Map'
import { MapContainer, Marker, Rectangle, TileLayer, useMap, useMapEvents } from 'react-leaflet'
import L, { LatLngBoundsExpression } from 'leaflet'
import 'leaflet/dist/leaflet.css'
import { toYaml } from '../lib/yaml'

// Leaflet's default marker icons do not resolve under a bundler: point at the CDN copies.
const icon = new L.Icon({
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
  iconSize: [25, 41], iconAnchor: [12, 41],
})

type Point = [number, number]                      // [lat, lon]
type Area = [number, number, number, number]       // [lat_min, lon_min, lat_max, lon_max]
type Geometry = Point | Area
type Shape = 'point' | 'area'

// Leaflet throws on a coordinate that is not a number, and the throw happens while the map is
// rendering, which takes the whole page down. So a geometry counts as one only when every
// coordinate of it is a real number, and anything else is simply not drawn.
const finite = (value: unknown): number | undefined =>
  typeof value === 'number' && Number.isFinite(value) ? value : undefined
const allNumbers = (g: unknown, count: number): boolean =>
  Array.isArray(g) && g.length === count && g.every(item => finite(item) !== undefined)
const isArea = (g: unknown): g is Area => allNumbers(g, 4)
const isPoint = (g: unknown): g is Point => allNumbers(g, 2)
const round = (value: number) => +value.toFixed(3)

// Hold every geometry at the arity of the shape being edited: a point switched to an area becomes
// a box with no extent yet, so typing one corner of it can never leave a hole in the array.
function reshape(value: unknown, shape: Shape): Geometry {
  const g = Array.isArray(value) ? value : []
  const lat = finite(g[0]) ?? 0, lon = finite(g[1]) ?? 0
  if (shape === 'area') return [lat, lon, finite(g[2]) ?? lat, finite(g[3]) ?? lon]
  return isArea(value) ? [round((value[0] + value[2]) / 2), round((value[1] + value[3]) / 2)] : [lat, lon]
}

// A maximum typed before its minimum is still a box; the pipeline reads the corners in order, so
// the drawing sorts them and so does the saved value.
const ordered = (g: Geometry): Geometry => isArea(g)
  ? [Math.min(g[0], g[2]), Math.min(g[1], g[3]), Math.max(g[0], g[2]), Math.max(g[1], g[3])] : g
const areaBounds = (a: Area): LatLngBoundsExpression =>
  [[Math.min(a[0], a[2]), Math.min(a[1], a[3])], [Math.max(a[0], a[2]), Math.max(a[1], a[3])]]

// Shapes handled:
//   coordinates:  [lat, lon] | [lat_min, lon_min, lat_max, lon_max] | {place: [...]}   (kind "coordinates")
//   area_group:   [[lat_min, lon_min], [lat_max, lon_max]]                                (kind "area_group")
//   image_coord:  {api: [lat_max, lat_min, lon_min, lon_max]} (legacy order; edited as YAML with the map as a guide)
function toGeometries(kind: string, value: any): Record<string, Geometry> {
  if (kind === 'area_group') {
    const corners = Array.isArray(value) && value.length === 2 && Array.isArray(value[0]) && Array.isArray(value[1])
      ? [value[0][0], value[0][1], value[1][0], value[1][1]] : null
    return isArea(corners) ? { area: corners } : {}
  }
  if (isPoint(value) || isArea(value)) return { site: value as Geometry }
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    const out: Record<string, Geometry> = {}
    for (const [k, v] of Object.entries(value)) if (isPoint(v) || isArea(v)) out[k] = v as Geometry
    return out
  }
  return {}
}

function fromGeometries(kind: string, geoms: Record<string, Geometry>, manyPlaces: boolean): any {
  // Only complete geometries are saved, and areas are saved corner-min first: the schema takes
  // [lat, lon] or [lat_min, lon_min, lat_max, lon_max], and the NWP reader selects the grid
  // between the two corners in that order.
  const complete: Record<string, Geometry> = {}
  for (const [name, g] of Object.entries(geoms)) if (isPoint(g) || isArea(g)) complete[name] = ordered(g)
  if (kind === 'area_group') {
    const a = complete.area
    return isArea(a) ? [[a[0], a[1]], [a[2], a[3]]] : undefined
  }
  if (manyPlaces) return complete
  return Object.values(complete)[0]
}

function ClickHandler({ onClick }: { onClick: (lat: number, lon: number) => void }) {
  useMapEvents({ click: e => onClick(+e.latlng.lat.toFixed(3), +e.latlng.lng.toFixed(3)) })
  return null
}

// The map is mounted once and told where to look. Keying it on the centre instead rebuilt the map
// and reloaded every tile on each keystroke, and following the centre as it is typed would drag the
// view out from under the hand that is typing; so the view moves when the place being edited
// changes, and holds still while its numbers are edited.
function View({ center, zoom, place }: { center: Point; zoom: number; place: string }) {
  const map = useMap()
  const latest = useRef({ center, zoom })
  latest.current = { center, zoom }
  useEffect(() => { map.setView(latest.current.center, latest.current.zoom) }, [map, place])
  return null
}

export default function MapField(props: FieldProps) {
  const { formData, onChange, name, schema } = props
  const kind = name === 'area_group' ? 'area_group' : name === 'coordinates' ? 'coordinates' : 'other'
  const [open, setOpen] = useState(false)
  const summary = formData === undefined || formData === null ? 'not set' : toYaml(formData).replace(/\n/g, ' ').trim()
  if (kind === 'other') {
    // image_coord and the like: no fixed geometry; keep the YAML box behaviour with a note
    return <TextField fullWidth size="small" sx={{ mb: 1.5 }} label={schema.title ?? name} value={summary} InputProps={{ readOnly: true }}
      helperText="edited on the Experiments/YAML side for now" />
  }
  return (
    <Box sx={{ mb: 1.5 }}>
      <Stack direction="row" spacing={1} alignItems="center">
        <TextField fullWidth size="small" label={schema.title ?? name} value={summary} InputProps={{ readOnly: true }} />
        <Button variant="outlined" startIcon={<MapIcon />} onClick={() => setOpen(true)} sx={{ whiteSpace: 'nowrap' }}>Map</Button>
      </Stack>
      {open && <MapDialog kind={kind} value={formData} onClose={() => setOpen(false)} onSave={v => { onChange(v); setOpen(false) }} />}
    </Box>
  )
}

function MapDialog({ kind, value, onClose, onSave }: { kind: string; value: any; onClose: () => void; onSave: (v: any) => void }) {
  const initial = useMemo(() => toGeometries(kind, value), [kind, value])
  const [geoms, setGeoms] = useState<Record<string, Geometry>>(initial)
  const [manyPlaces, setManyPlaces] = useState(value && typeof value === 'object' && !Array.isArray(value))
  const [shape, setShape] = useState<Shape>(kind === 'area_group' || Object.values(initial).some(isArea) ? 'area' : 'point')
  const [current, setCurrent] = useState<string>(Object.keys(initial)[0] ?? (kind === 'area_group' ? 'area' : 'site'))
  const [corner, setCorner] = useState<Point | null>(null)
  const [newName, setNewName] = useState('')
  useEffect(() => { if (kind === 'area_group') { setShape('area'); setCurrent('area'); setManyPlaces(false) } }, [kind])

  const names = Object.keys(geoms)
  const active = geoms[current]
  const center: Point = isPoint(active) ? active : isArea(active) ? [(active[0] + active[2]) / 2, (active[1] + active[3]) / 2] : [38.0, 23.7]

  const onMapClick = (lat: number, lon: number) => {
    if (shape === 'point') { setGeoms({ ...geoms, [current]: [lat, lon] }); return }
    if (!corner) { setCorner([lat, lon]); return }
    const area: Area = [Math.min(corner[0], lat), Math.min(corner[1], lon), Math.max(corner[0], lat), Math.max(corner[1], lon)]
    setGeoms({ ...geoms, [current]: area }); setCorner(null)
  }
  const setNumber = (index: number, text: string) => {
    const n = parseFloat(text); if (Number.isNaN(n)) return
    const g = reshape(active, shape) as number[]
    g[index] = n; setGeoms({ ...geoms, [current]: g as Geometry })
  }
  // The schema asks every place of one case to be the same shape, so the switch converts them all.
  const changeShape = (next: Shape) => {
    setShape(next); setCorner(null)
    setGeoms(Object.fromEntries(Object.entries(geoms).map(([name, g]) => [name, reshape(g, next)])))
  }
  const labels = shape === 'point' ? ['lat', 'lon'] : ['lat min', 'lon min', 'lat max', 'lon max']

  return (
    <Dialog open onClose={onClose} fullWidth maxWidth="md">
      <DialogTitle>{kind === 'area_group' ? 'Area of the NWP grid' : 'Site coordinates'}</DialogTitle>
      <DialogContent dividers>
        <Stack spacing={1.5}>
          <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap">
            {kind !== 'area_group' && (
              <>
                <ToggleButtonGroup size="small" exclusive value={shape} onChange={(_, v) => v && changeShape(v)}>
                  <ToggleButton value="point">point</ToggleButton><ToggleButton value="area">area</ToggleButton>
                </ToggleButtonGroup>
                <FormControlLabel control={<Switch size="small" checked={manyPlaces} onChange={e => setManyPlaces(e.target.checked)} />} label="several places" />
              </>
            )}
            {manyPlaces && (
              <Stack direction="row" spacing={1} alignItems="center">
                <ToggleButtonGroup size="small" exclusive value={current} onChange={(_, v) => v && setCurrent(v)}>
                  {names.map(n => <ToggleButton key={n} value={n}>{n}</ToggleButton>)}
                </ToggleButtonGroup>
                <TextField size="small" placeholder="new place" value={newName} onChange={e => setNewName(e.target.value)} sx={{ width: 140 }} />
                <Button size="small" onClick={() => { const n = newName.trim(); if (n) { setGeoms({ ...geoms, [n]: shape === 'point' ? center : [center[0] - 0.5, center[1] - 0.5, center[0] + 0.5, center[1] + 0.5] }); setCurrent(n); setNewName('') } }}>add</Button>
              </Stack>
            )}
          </Stack>
          <Stack direction="row" spacing={1}>
            {labels.map((l, i) => <TextField key={l} size="small" label={l} type="number" inputProps={{ step: 0.01 }}
              value={active?.[i] ?? ''} onChange={e => setNumber(i, e.target.value)} sx={{ width: 130 }} />)}
          </Stack>
          <Typography variant="caption" color="text.secondary">
            {shape === 'point' ? 'Click the map to place the site.' : corner ? 'Click the opposite corner of the area.' : 'Click two opposite corners to draw the area.'}
          </Typography>
          <Box sx={{ height: 420, borderRadius: 1, overflow: 'hidden' }}>
            <MapContainer center={center} zoom={isArea(active) ? 6 : 8} style={{ height: '100%', width: '100%' }}>
              <TileLayer attribution='&copy; OpenStreetMap contributors' url="https://tile.openstreetmap.org/{z}/{x}/{y}.png" />
              <View center={center} zoom={isArea(active) ? 6 : 8} place={`${current}:${shape}`} />
              <ClickHandler onClick={onMapClick} />
              {Object.entries(geoms).map(([n, g]) => isPoint(g)
                ? <Marker key={n} position={g} icon={icon} />
                : isArea(g)
                  ? <Rectangle key={n} bounds={areaBounds(g)} pathOptions={{ color: n === current ? '#1f5fbf' : '#888' }} />
                  : null)}
              {corner && <Marker position={corner} icon={icon} opacity={0.5} />}
            </MapContainer>
          </Box>
          {kind === 'area_group' && <Alert severity="info">Latitudes and longitudes of the area must be multiples of the NWP resolution (0.1° for ECMWF).</Alert>}
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button variant="contained" onClick={() => onSave(fromGeometries(kind, geoms, manyPlaces))}>Use</Button>
      </DialogActions>
    </Dialog>
  )
}
