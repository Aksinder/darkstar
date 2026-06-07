/**
 * PowerFlowRegistry.ts
 *
 * Defines the extensible node registry for the PowerFlowCard.
 * Nodes can be enabled/disabled based on system configuration.
 */

import { Sun, Home, Battery, BatteryCharging, Zap, Droplets, Car, Plug } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'

export interface PowerFlowData {
    solar: { kw: number; todayKwh?: number }
    battery: { kw: number; soc: number } // +charge, -discharge
    grid: { kw: number; importKwh?: number; exportKwh?: number } // +import, -export
    house: { kw: number; todayKwh?: number }
    water: { kw: number; todayKwh?: number }
    ev?: { kw: number }
    evPluggedIn?: boolean // Rev UI18: Streamed from backend
    evSoc?: number // Rev F50 Phase 5: EV battery SoC percentage
    evChargers?: Array<{ name: string; kw: number; soc: number | null; pluggedIn: boolean }> // Rev F64: Per-EV details
}

export interface FlowNodeConfig {
    id: 'solar' | 'house' | 'battery' | 'grid' | 'water' | 'ev'
    configKey?: string // Required for optional nodes (e.g., 'system.has_solar')
    lucideIcon: LucideIcon | ((data: PowerFlowData) => LucideIcon)
    lucideIconCharging?: LucideIcon
    color: string | ((data: PowerFlowData) => string)
    label: string | ((data: PowerFlowData) => string)
    valueAccessor: (data: PowerFlowData) => string
    subValueAccessor?: (data: PowerFlowData) => string | undefined
    glowIntensityAccessor: (data: PowerFlowData) => number
    isChargingAccessor?: (data: PowerFlowData) => boolean
    shouldRender?: (data: PowerFlowData, configMap: Record<string, unknown> | null) => boolean // Rev UI18
}

// =============================================================================
// HELPERS
// =============================================================================

const fmtKw = (v: number) => {
    const absV = Math.abs(v)
    if (absV > 0 && absV < 0.1) return `${absV.toFixed(2)} kW`
    return `${absV.toFixed(1)} kW`
}

const fmtKwh = (v?: number) => (v != null ? `${v.toFixed(1)} kWh` : undefined)

// =============================================================================
// REGISTRY
// =============================================================================

export const NODE_REGISTRY: FlowNodeConfig[] = [
    {
        id: 'solar',
        configKey: 'system.has_solar',
        lucideIcon: Sun,
        color: 'rgb(var(--color-accent))',
        label: 'Solar',
        valueAccessor: (data) => fmtKw(data.solar.kw),
        subValueAccessor: (data) => fmtKwh(data.solar.todayKwh),
        glowIntensityAccessor: (data) => Math.min(data.solar.kw / 6, 1),
    },
    {
        id: 'house',
        lucideIcon: Home,
        color: 'rgb(var(--color-house))',
        label: 'House',
        valueAccessor: (data) => fmtKw(data.house.kw),
        subValueAccessor: (data) => fmtKwh(data.house.todayKwh),
        glowIntensityAccessor: (data) => Math.min(data.house.kw / 6, 1),
    },
    {
        id: 'battery',
        configKey: 'system.has_battery',
        lucideIcon: Battery,
        lucideIconCharging: BatteryCharging,
        color: 'rgb(var(--color-good))',
        label: (data) => (data.battery.kw <= 0 ? 'Charge' : 'Discharge'),
        valueAccessor: (data) => fmtKw(data.battery.kw),
        subValueAccessor: (data) => `${data.battery.soc.toFixed(0)}%`,
        glowIntensityAccessor: (data) => Math.min(Math.abs(data.battery.kw) / 6, 1),
        isChargingAccessor: (data) => data.battery.kw < 0,
    },
    {
        id: 'grid',
        lucideIcon: Zap,
        color: 'rgb(var(--color-grid))',
        label: (data) => (data.grid.kw >= 0 ? 'Import' : 'Export'),
        valueAccessor: (data) => fmtKw(data.grid.kw),
        subValueAccessor: (data) => (data.grid.kw >= 0 ? fmtKwh(data.grid.importKwh) : fmtKwh(data.grid.exportKwh)),
        glowIntensityAccessor: (data) => Math.min(Math.abs(data.grid.kw) / 6, 1),
    },
    {
        id: 'water',
        configKey: 'system.has_water_heater',
        lucideIcon: Droplets,
        color: 'rgb(var(--color-water))',
        label: 'Water',
        valueAccessor: (data) => fmtKw(data.water.kw),
        subValueAccessor: (data) => fmtKwh(data.water.todayKwh),
        glowIntensityAccessor: (data) => Math.min(data.water.kw / 6, 1),
    },
    {
        id: 'ev',
        configKey: 'system.has_ev_charger', // Match settings key
        // Only render the EV node when a car is plugged in AT HOME. The backend home-gates the
        // live status (an away car charging at a Supercharger still reads plugged), so evPluggedIn
        // is false when the car is away -> don't imply we're charging a car that isn't here.
        shouldRender: (data: PowerFlowData) => data.evPluggedIn === true,
        lucideIcon: (data: PowerFlowData) => (data.evPluggedIn ? Plug : Car),
        color: (data: PowerFlowData) => (data.evPluggedIn ? 'rgb(var(--color-ai))' : 'rgb(var(--color-muted))'),
        label: 'EV',
        valueAccessor: (data) => (data.ev ? fmtKw(data.ev.kw) : '0.0 kW'),
        subValueAccessor: (data) => {
            // Phase 7: Show SoC of plugged-in EV, or first EV if multiple plugged in
            if (!data.evChargers || data.evChargers.length === 0) return undefined
            const pluggedIn = data.evChargers.filter((ev) => ev.pluggedIn)
            const evToShow = pluggedIn.length > 0 ? pluggedIn[0] : data.evChargers[0]
            return evToShow?.soc != null ? `${evToShow.soc.toFixed(0)}%` : '--%'
        },
        glowIntensityAccessor: (data) => (data.ev ? Math.min(data.ev.kw / 11, 1) : 0),
    },
]
