import React, { useEffect, useMemo, useState } from 'react'
import { Reorder } from 'framer-motion'
import { Api, ControllableLoad } from '../../../lib/api'

type Assignment = { tier: string; rank: number }
type Assignments = Record<string, Assignment>

const TIERS: { id: string; label: string; hint: string }[] = [
    { id: 'important', label: 'Important', hint: 'runs almost always' },
    { id: 'comfort', label: 'Comfort', hint: 'only when cheap / surplus' },
]

interface Props {
    value: string
    onChange: (loads: Assignments) => void
    disabled?: boolean
}

function parseAssignments(value: string): Assignments {
    try {
        const parsed: unknown = JSON.parse(value || '{}')
        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
            const out: Assignments = {}
            for (const [id, v] of Object.entries(parsed as Record<string, unknown>)) {
                if (v && typeof v === 'object') {
                    const rec = v as Record<string, unknown>
                    const tier = String(rec.tier ?? '')
                    const rank = Number(rec.rank ?? 0)
                    if (tier) out[id] = { tier, rank: Number.isFinite(rank) ? rank : 0 }
                }
            }
            return out
        }
    } catch {
        /* ignore malformed value */
    }
    return {}
}

export const LoadPriorityEditor: React.FC<Props> = ({ value, onChange, disabled }) => {
    const [loads, setLoads] = useState<ControllableLoad[]>([])
    const [loadError, setLoadError] = useState<string | null>(null)
    const assignments = useMemo(() => parseAssignments(value), [value])

    useEffect(() => {
        let alive = true
        Api.controllableLoads()
            .then((r) => {
                if (alive) setLoads(r.loads || [])
            })
            .catch(() => {
                if (alive) setLoadError('Could not load the list of controllable loads.')
            })
        return () => {
            alive = false
        }
    }, [])

    const loadMeta = useMemo(() => {
        const m: Record<string, ControllableLoad> = {}
        loads.forEach((l) => {
            m[l.id] = l
        })
        return m
    }, [loads])

    // Union of API-reported loads and any already-assigned ids (so stale assignments still show).
    const allIds = useMemo(() => {
        const s = new Set<string>(loads.map((l) => l.id))
        Object.keys(assignments).forEach((id) => s.add(id))
        return Array.from(s)
    }, [loads, assignments])

    const idsInTier = (tier: string): string[] =>
        allIds
            .filter((id) => (assignments[id]?.tier ?? '') === tier)
            .sort((a, b) => (assignments[a]?.rank ?? 0) - (assignments[b]?.rank ?? 0))

    const unassignedIds = allIds.filter((id) => !assignments[id]?.tier)

    const moveToTier = (id: string, tier: string) => {
        const next: Assignments = { ...assignments }
        if (!tier) {
            delete next[id]
        } else {
            next[id] = { tier, rank: idsInTier(tier).length }
        }
        onChange(next)
    }

    const reorderTier = (tier: string, orderedIds: string[]) => {
        const next: Assignments = { ...assignments }
        orderedIds.forEach((id, i) => {
            next[id] = { tier, rank: i }
        })
        onChange(next)
    }

    const typeLabel = (id: string) => (loadMeta[id]?.type ?? '').replace('_', ' ')

    return (
        <div className="space-y-4">
            {loadError && <div className="text-xs text-bad">{loadError}</div>}

            {TIERS.map((tier) => {
                const ids = idsInTier(tier.id)
                return (
                    <div key={tier.id} className="rounded-ds-lg border border-line/40 bg-surface p-3">
                        <div className="mb-2 flex items-baseline justify-between">
                            <span className="text-sm font-semibold text-text">{tier.label}</span>
                            <span className="text-[11px] text-muted">{tier.hint}</span>
                        </div>
                        {ids.length === 0 ? (
                            <div className="rounded-ds-lg border border-dashed border-line/40 px-3 py-2 text-[11px] text-muted">
                                No loads in this tier — assign one below.
                            </div>
                        ) : (
                            <Reorder.Group
                                axis="y"
                                values={ids}
                                onReorder={(newOrder: string[]) => reorderTier(tier.id, newOrder)}
                                className="space-y-2"
                            >
                                {ids.map((id, i) => (
                                    <Reorder.Item
                                        key={id}
                                        value={id}
                                        className="flex cursor-grab items-center justify-between gap-2 rounded-ds-lg border border-line/40 bg-base px-3 py-2 text-sm hover:bg-accent/20 active:cursor-grabbing"
                                    >
                                        <span className="flex items-center gap-2">
                                            <span className="tabular-nums text-[11px] text-muted">#{i + 1}</span>
                                            <span className="font-medium text-text">{loadMeta[id]?.name ?? id}</span>
                                            <span className="text-[10px] uppercase text-muted">{typeLabel(id)}</span>
                                        </span>
                                        <select
                                            value={tier.id}
                                            disabled={disabled}
                                            onPointerDown={(e) => e.stopPropagation()}
                                            onChange={(e) => moveToTier(id, e.target.value)}
                                            className="rounded-ds-lg border border-line/40 bg-surface px-1.5 py-1 text-[11px] text-text"
                                        >
                                            {TIERS.map((t) => (
                                                <option key={t.id} value={t.id}>
                                                    {t.label}
                                                </option>
                                            ))}
                                            <option value="">Unassign</option>
                                        </select>
                                    </Reorder.Item>
                                ))}
                            </Reorder.Group>
                        )}
                    </div>
                )
            })}

            <div className="rounded-ds-lg border border-dashed border-line/40 bg-surface p-3">
                <div className="mb-2 text-sm font-semibold text-muted">Unassigned</div>
                {unassignedIds.length === 0 ? (
                    <div className="text-[11px] text-muted">All known loads are assigned.</div>
                ) : (
                    <div className="space-y-2">
                        {unassignedIds.map((id) => (
                            <div
                                key={id}
                                className="flex items-center justify-between gap-2 rounded-ds-lg border border-line/40 bg-base px-3 py-2 text-sm"
                            >
                                <span className="flex items-center gap-2">
                                    <span className="font-medium text-text">{loadMeta[id]?.name ?? id}</span>
                                    <span className="text-[10px] uppercase text-muted">{typeLabel(id)}</span>
                                </span>
                                <select
                                    value=""
                                    disabled={disabled}
                                    onChange={(e) => moveToTier(id, e.target.value)}
                                    className="rounded-ds-lg border border-line/40 bg-surface px-1.5 py-1 text-[11px] text-text"
                                >
                                    <option value="">Assign to…</option>
                                    {TIERS.map((t) => (
                                        <option key={t.id} value={t.id}>
                                            {t.label}
                                        </option>
                                    ))}
                                </select>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            <p className="text-[11px] text-muted">
                Drag cards within a tier to set priority order (top = highest). Use the dropdown to move a load between
                tiers. Loads only respond to priority when they have a daily need / pending run.
            </p>
        </div>
    )
}
