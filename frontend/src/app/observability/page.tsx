"use client";

import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
    ArrowRight, Search, Zap, Clock, Code, Activity, ChevronRight, Plus
} from "lucide-react";
import { useRouter } from "next/navigation";
import { Sidebar } from "@/components/Sidebar";
import { createClient } from "@/utils/supabase";

interface AnalyticsRow {
    id: string;
    model: string;
    tokens_in: number;
    tokens_saved: number;
    latency_ms: number;
    timestamp: number;
}

export default function ObservabilityListPage() {
    const router = useRouter();
    const [requests, setRequests] = useState<AnalyticsRow[]>([]);
    const [displayLimit, setDisplayLimit] = useState(10);
    const [loading, setLoading] = useState(true);

    const GATEWAY_URL = process.env.NEXT_PUBLIC_GATEWAY_URL || "http://localhost:8000";
    const supabase = createClient();

    useEffect(() => {
        const fetchRequests = async () => {
            const { data: { user } } = await supabase.auth.getUser();
            const userId = user?.id || (process.env.NODE_ENV === 'development' ? 'dev-user' : null);

            if (userId) {
                try {
                    const resp = await fetch(`${GATEWAY_URL}/api/user/stats`, {
                        headers: { "x-user-id": userId }
                    });
                    const data = await resp.json();
                    const fetched = (data.recent_requests || []);

                    if (fetched.length === 0 && process.env.NODE_ENV === 'development') {
                        // Inject Mock Requests for Local Test
                        setRequests([
                            { id: 'mock-1', model: 'anthropic/claude-3-sonnet', tokens_in: 5200, tokens_saved: 4100, latency_ms: 1840, timestamp: Date.now() / 1000 - 3600 },
                            { id: 'mock-2', model: 'openai/gpt-4-turbo', tokens_in: 8400, tokens_saved: 6200, latency_ms: 3200, timestamp: Date.now() / 1000 - 7200 },
                            { id: 'mock-3', model: 'meta-llama/llama-3-70b', tokens_in: 2100, tokens_saved: 1500, latency_ms: 950, timestamp: Date.now() / 1000 - 10800 },
                            { id: 'mock-4', model: 'google/gemini-pro-1.5', tokens_in: 12500, tokens_saved: 9800, latency_ms: 4500, timestamp: Date.now() / 1000 - 14400 },
                        ]);
                    } else {
                        setRequests(fetched.sort((a: any, b: any) => b.timestamp - a.timestamp));
                    }
                } catch (e) {
                    console.error("Failed to fetch requests", e);
                }
                setLoading(false);
            }
        };
        fetchRequests();
    }, [supabase.auth, GATEWAY_URL]);

    const loadMore = () => setDisplayLimit(prev => prev + 10);

    return (
        <div className="layout-wrapper">
            <Sidebar />
            <main className="main-content">

                <header style={{ marginBottom: "48px" }}>
                    <h1 style={{ fontSize: "24px", fontWeight: 700, letterSpacing: '-0.03em' }}>Deep Observability</h1>
                    <p className="text-muted" style={{ fontSize: "14px", marginTop: "4px" }}>Select a request to step into the reconstruction observer.</p>
                </header>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', maxWidth: '1000px' }}>
                    <AnimatePresence>
                        {requests.slice(0, displayLimit).map((req, idx) => (
                            <motion.div
                                key={req.id}
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: idx * 0.03 }}
                                onClick={() => router.push(`/analytics/${req.id}`)}
                                className="glass card"
                                style={{
                                    padding: '20px 24px',
                                    display: 'grid',
                                    gridTemplateColumns: '1fr 1fr 1fr 1fr 40px',
                                    alignItems: 'center',
                                    cursor: 'pointer',
                                    transition: 'all 0.2s cubic-bezier(0.16, 1, 0.3, 1)',
                                    border: '1px solid rgba(255,255,255,0.05)'
                                }}
                                whileHover={{ scale: 1.005, borderColor: 'var(--accent)', background: 'rgba(255,255,255,0.03)' }}
                            >
                                <div>
                                    <div style={{ fontSize: '10px', fontWeight: 800, opacity: 0.3, marginBottom: '4px', letterSpacing: '0.05em' }}>TIMESTAMP</div>
                                    <div style={{ fontSize: '13px', fontWeight: 500 }}>{new Date(req.timestamp * 1000).toLocaleString()}</div>
                                </div>

                                <div>
                                    <div style={{ fontSize: '10px', fontWeight: 800, opacity: 0.3, marginBottom: '4px', letterSpacing: '0.05em' }}>MODEL</div>
                                    <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--accent)' }}>{req.model.split('/').pop()}</div>
                                </div>

                                <div>
                                    <div style={{ fontSize: '10px', fontWeight: 800, opacity: 0.3, marginBottom: '4px', letterSpacing: '0.05em' }}>OPTIMIZATION</div>
                                    <div style={{ fontSize: '13px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                                        <Zap size={12} className="text-accent" />
                                        <span>{((req.tokens_saved / (req.tokens_in + req.tokens_saved || 1)) * 100).toFixed(1)}%</span>
                                    </div>
                                </div>

                                <div>
                                    <div style={{ fontSize: '10px', fontWeight: 800, opacity: 0.3, marginBottom: '4px', letterSpacing: '0.05em' }}>LATENCY</div>
                                    <div style={{ fontSize: '13px', opacity: 0.8 }}>{req.latency_ms.toFixed(0)}ms</div>
                                </div>

                                <div style={{ display: 'flex', justifyContent: 'flex-end', opacity: 0.2 }}>
                                    <ChevronRight size={18} />
                                </div>
                            </motion.div>
                        ))}
                    </AnimatePresence>

                    {requests.length > displayLimit && (
                        <motion.button
                            onClick={loadMore}
                            whileHover={{ scale: 1.02, background: 'rgba(255,255,255,0.08)' }}
                            className="glass"
                            style={{
                                marginTop: '12px',
                                padding: '16px',
                                borderRadius: '16px',
                                border: '1px dashed rgba(255,255,255,0.1)',
                                color: 'rgba(255,255,255,0.4)',
                                fontSize: '13px',
                                fontWeight: 600,
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                gap: '8px',
                                cursor: 'pointer'
                            }}
                        >
                            <Plus size={14} /> Load 10 More Requests
                        </motion.button>
                    )}

                    {!loading && requests.length === 0 && (
                        <div style={{ textAlign: 'center', padding: '100px 0', opacity: 0.2 }}>
                            <Search size={48} style={{ margin: '0 auto 16px' }} />
                            <p>No optimization trails found yet.</p>
                        </div>
                    )}
                </div>
            </main>
        </div>
    );
}
