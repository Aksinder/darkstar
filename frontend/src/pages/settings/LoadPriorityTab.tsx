import React from 'react'
import { useNavigate } from 'react-router-dom'
import Card from '../../components/Card'
import { useSettingsForm } from './hooks/useSettingsForm'
import { SettingsField } from './components/SettingsField'
import { loadPriorityFieldList, loadPrioritySections } from './types'
import { shouldRenderField } from './logic'
import { UnsavedChangesBanner } from './components/UnsavedChangesBanner'
import { NavigationBlockerDialog } from './components/NavigationBlockerDialog'
import { useUnsavedChangesGuard } from './hooks/useUnsavedChangesGuard'

export const LoadPriorityTab: React.FC<{ advancedMode?: boolean }> = () => {
    const navigate = useNavigate()
    const {
        config,
        form,
        fieldErrors,
        loading,
        saving,
        statusMessage,
        handleChange,
        save,
        isDirty,
        haEntities,
        haLoading,
    } = useSettingsForm(loadPriorityFieldList, [])

    const blocker = useUnsavedChangesGuard(isDirty)

    if (loading) {
        return <Card className="p-6 text-sm text-muted">Loading load priority configuration…</Card>
    }

    return (
        <div className="space-y-4">
            <UnsavedChangesBanner visible={isDirty} onSave={() => save()} saving={saving} />

            {loadPrioritySections.map((section) => (
                <Card key={section.title} className="overflow-hidden">
                    <div className="px-6 py-4">
                        <h2 className="text-lg font-semibold text-foreground">{section.title}</h2>
                        {section.description && <p className="mt-1 text-sm text-muted">{section.description}</p>}
                    </div>
                    <div className="grid grid-cols-1 gap-4 p-6 md:grid-cols-2">
                        {section.fields.map((field) => {
                            if (!shouldRenderField(field, form, config as Record<string, unknown>)) return null
                            return (
                                <SettingsField
                                    key={field.key}
                                    field={field}
                                    value={form[field.key]}
                                    onChange={handleChange}
                                    error={fieldErrors[field.key]}
                                    haEntities={haEntities}
                                    haLoading={haLoading}
                                    fullForm={form}
                                    config={config as Record<string, unknown>}
                                />
                            )
                        })}
                    </div>
                </Card>
            ))}

            <div className="flex flex-wrap items-center gap-3">
                <button
                    disabled={saving}
                    onClick={() => save()}
                    className="flex items-center justify-center gap-2 rounded-xl px-3 py-2.5 text-[11px] font-semibold transition btn-glow-primary bg-accent hover:bg-accent2 text-[#100f0e] disabled:opacity-50"
                >
                    {saving ? 'Saving…' : 'Save Load Priority'}
                </button>
                {statusMessage && (
                    <div
                        className={`rounded-lg p-3 text-sm ${
                            statusMessage.startsWith('Please fix') ||
                            statusMessage.startsWith('Save failed') ||
                            statusMessage.startsWith('Failed to load')
                                ? 'bg-bad/10 border border-bad/30 text-bad'
                                : 'bg-good/10 border border-good/30 text-good'
                        }`}
                    >
                        {statusMessage}
                    </div>
                )}
            </div>

            <NavigationBlockerDialog
                visible={blocker.state === 'blocked'}
                onStay={() => blocker.reset?.()}
                onLeave={() => {
                    if (blocker.location) {
                        navigate(blocker.location.pathname + blocker.location.search, {
                            state: { ...blocker.location.state, ignoreUnsavedChangesGuard: true },
                        })
                    }
                }}
            />
        </div>
    )
}
