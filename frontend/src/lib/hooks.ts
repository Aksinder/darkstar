import { useEffect, useRef, useSyncExternalStore } from 'react'
import { getConnectionState, getSocket, subscribeConnection, type ConnectionState } from './socket'

export const useSocket = (event: string, callback: (data: unknown) => void) => {
    // Store the latest callback in a ref so we don't have to re-subscribe on every render
    const callbackRef = useRef(callback)

    // Update ref when callback changes
    useEffect(() => {
        callbackRef.current = callback
    }, [callback])

    useEffect(() => {
        const socket = getSocket()

        // Use a stable wrapper function
        const handleEvent = (data: unknown) => {
            if (callbackRef.current) {
                callbackRef.current(data)
            }
        }

        socket.on(event, handleEvent)

        return () => {
            socket.off(event, handleEvent)
        }
    }, [event]) // Only re-subscribe if the event name itself changes
}

/** Tri-state socket liveness. Eagerly initialises the socket singleton. */
const subscribeAndInit = (onChange: () => void) => {
    getSocket()
    return subscribeConnection(onChange)
}

export const useConnectionState = (): ConnectionState => useSyncExternalStore(subscribeAndInit, getConnectionState)

/**
 * Calls `onReconnect` whenever the socket comes back after a drop (not on the
 * first connect). Pushes emitted while disconnected are lost, so callers
 * should refetch their state here.
 */
export const useOnReconnect = (onReconnect: () => void) => {
    const state = useConnectionState()
    const callbackRef = useRef(onReconnect)
    const hasConnected = useRef(false)

    useEffect(() => {
        callbackRef.current = onReconnect
    }, [onReconnect])

    useEffect(() => {
        if (state !== 'connected') return
        if (hasConnected.current) callbackRef.current()
        hasConnected.current = true
    }, [state])
}
