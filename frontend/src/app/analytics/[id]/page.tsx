"use client";

import { useState, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
import { Sidebar } from "@/components/Sidebar";
import { motion } from "framer-motion";
import {
    ArrowLeft, Shield, Database, Activity, Cpu, Server, Zap
} from "lucide-react";
import { ReconstructionObserver } from "@/components/ReconstructionObserver";
import { createClient } from "@/utils/supabase";

export default function AnalyticsDetailPage() {
    const params = useParams();
    const router = useRouter();
    const [requestData, setRequestData] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const supabase = createClient();
    const [relatedRequests, setRelatedRequests] = useState<any[]>([]);

    const GATEWAY_URL = process.env.NEXT_PUBLIC_GATEWAY_URL || "http://localhost:8000";

    useEffect(() => {
        const fetchDetail = async () => {
            const { data: { user } } = await supabase.auth.getUser();
            const userId = user?.id || (process.env.NODE_ENV === 'development' ? 'dev-user' : null);

            if (userId) {
                try {
                    const resp = await fetch(`${GATEWAY_URL}/api/user/stats`, {
                        headers: { "x-user-id": userId }
                    });
                    const data = await resp.json();
                    const allReqs = data.recent_requests || [];

                    // Priority 1: Match exactly by analytics ID
                    // Priority 2: Match by session_id (returns the latest one in that thread)
                    let req = allReqs.find((r: any) => r.id === params.id);
                    if (!req) {
                        req = allReqs.find((r: any) => r.session_id === params.id);
                    }

                    if (req) {
                        setRequestData(req);
                        // Filter for other requests in the same fingerprint thread (session_id)
                        const thread = allReqs.filter((r: any) => r.session_id === req.session_id && r.id !== req.id);
                        setRelatedRequests(thread.sort((a: any, b: any) => b.timestamp - a.timestamp));
                    }
                } catch (e) {
                    console.error("Failed to fetch detail", e);
                }
            }
            setLoading(false);
        };
        fetchDetail();
    }, [params.id, GATEWAY_URL, supabase.auth]);

    // Robust Fallback Access using real DB column 'metadata'
    const or_data = requestData?.metadata || {};
    const usage_cost = or_data.total_cost || requestData?.total_cost_usd || 0;

    // Formatting Helpers
    const formatCost = (val: number) => `$${(val || 0).toFixed(6)}`;
    const formatMs = (val: number) => `${Math.round(val || 0)}ms`;

    if (loading) return <div style={{ background: '#050505', minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white' }}>LOADING SCAN...</div>;

    return (
        <div className="layout-wrapper" style={{ background: '#050505', minHeight: '100vh' }}>
            <Sidebar />
            <main className="main-content" style={{ padding: '64px' }}>

                {/* Navigation & Header */}
                <header style={{ marginBottom: "64px" }}>
                    <button
                        onClick={() => router.push('/analytics')}
                        style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px',
                            fontSize: '11px',
                            background: 'none',
                            border: 'none',
                            padding: 0,
                            cursor: 'pointer',
                            marginBottom: '24px',
                            fontWeight: 800,
                            letterSpacing: '0.1em',
                            color: 'rgba(255,255,255,0.4)'
                        }}
                    >
                        <ArrowLeft size={12} /> BACK TO DASHBOARD
                    </button>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
                                <h1 style={{ fontSize: "36px", fontWeight: 700, letterSpacing: '-0.04em', lineHeight: 1, margin: 0 }}>Optimization Report</h1>
                                <span style={{
                                    background: 'rgba(var(--accent-rgb), 0.15)',
                                    color: 'var(--accent)',
                                    fontSize: '10px',
                                    fontWeight: 800,
                                    padding: '4px 10px',
                                    borderRadius: '100px',
                                    border: '1px solid rgba(var(--accent-rgb), 0.2)',
                                    letterSpacing: '0.05em'
                                }}>
                                    THREAD ISOLATED
                                </span>
                            </div>
                            <div style={{ fontFamily: 'monospace', fontSize: '13px', color: 'rgba(255,255,255,0.3)', display: 'flex', gap: '16px' }}>
                                <span>FINGERPRINT: <span style={{ color: 'rgba(255,255,255,0.6)' }}>{requestData?.session_id || 'N/A'}</span></span>
                                {or_data.id && <span>OR_ID: {or_data.id}</span>}
                            </div>
                        </div>
                        <div style={{ textAlign: 'right' }}>
                            <div style={{ fontSize: '11px', fontWeight: 800, letterSpacing: '0.1em', opacity: 0.4, marginBottom: '4px' }}>TOTAL COST</div>
                            <div style={{ fontSize: '24px', fontWeight: 700 }}>{formatCost(usage_cost)}</div>
                        </div>
                    </div>
                </header>

                {/* Primary Data Grid (Bento Style) */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '24px', marginBottom: '64px' }}>

                    {/* Financial Metrics */}
                    <div style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)', padding: '24px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '24px' }}>
                            <Shield size={14} className="text-accent" />
                            <span style={{ fontSize: '10px', fontWeight: 800, letterSpacing: '0.1em', opacity: 0.5 }}>COST BREAKDOWN</span>
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', fontSize: '12px' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span style={{ opacity: 0.5 }}>Upstream</span>
                                <span style={{ fontFamily: 'monospace' }}>{formatCost(or_data.upstream_inference_cost || 0)}</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--accent)' }}>
                                <span>Discount</span>
                                <span style={{ fontFamily: 'monospace' }}>-{formatCost(or_data.cache_discount || 0)}</span>
                            </div>
                            <div style={{ height: '1px', background: 'rgba(255,255,255,0.1)', margin: '4px 0' }} />
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontWeight: 700 }}>
                                <span>Total</span>
                                <span style={{ fontFamily: 'monospace' }}>{formatCost(usage_cost)}</span>
                            </div>
                        </div>
                    </div>

                    {/* Token Volumetrics */}
                    <div style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)', padding: '24px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '24px' }}>
                            <Database size={14} className="text-accent" />
                            <span style={{ fontSize: '10px', fontWeight: 800, letterSpacing: '0.1em', opacity: 0.5 }}>TOKEN USAGE</span>
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                            <div>
                                <div style={{ fontSize: '9px', opacity: 0.4, marginBottom: '4px' }}>PROMPT</div>
                                <div style={{ fontSize: '18px', fontWeight: 700 }}>{or_data.tokens_prompt || requestData?.tokens_in || 0}</div>
                            </div>
                            <div>
                                <div style={{ fontSize: '9px', opacity: 0.4, marginBottom: '4px' }}>COMPLETION</div>
                                <div style={{ fontSize: '18px', fontWeight: 700 }}>{or_data.tokens_completion || requestData?.tokens_out || 0}</div>
                            </div>
                            <div style={{ gridColumn: 'span 2' }}>
                                <div style={{ fontSize: '9px', opacity: 0.4, marginBottom: '4px' }}>SAVINGS</div>
                                <div style={{ fontSize: '18px', fontWeight: 700, color: 'var(--accent)' }}>{requestData?.tokens_saved || 0} T</div>
                            </div>
                        </div>
                    </div>

                    {/* Latency & Provider */}
                    <div style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)', padding: '24px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '24px' }}>
                            <Activity size={14} className="text-accent" />
                            <span style={{ fontSize: '10px', fontWeight: 800, letterSpacing: '0.1em', opacity: 0.5 }}>LATENCY</span>
                        </div>
                        <div style={{ marginBottom: '16px' }}>
                            <div style={{ fontSize: '9px', opacity: 0.4, marginBottom: '4px' }}>TOTAL ROUNDTRIP</div>
                            <div style={{ fontSize: '24px', fontWeight: 700 }}>{formatMs(or_data.latency || requestData?.latency_ms)}</div>
                        </div>
                        <div style={{ fontSize: '11px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', opacity: 0.6 }}>
                                <span>Optimization</span>
                                <span>{formatMs(requestData?.reconstruction_log?.overhead_ms)}</span>
                            </div>
                        </div>
                    </div>

                    {/* Model Details - Expanded */}
                    <div style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)', padding: '24px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '24px' }}>
                            <Cpu size={14} className="text-accent" />
                            <span style={{ fontSize: '10px', fontWeight: 800, letterSpacing: '0.1em', opacity: 0.5 }}>INFERENCE ENGINE</span>
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', fontSize: '11px' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <span style={{ opacity: 0.5 }}>Model ID</span>
                                <span style={{ textAlign: 'right', maxWidth: '120px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{requestData?.model}</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <span style={{ opacity: 0.5 }}>Route</span>
                                <span>{or_data.provider_name || 'Auto-Router'}</span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Animation: Sequence directly from DB */}
                <div style={{ marginBottom: '64px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px' }}>
                        <Zap size={18} className="text-accent" />
                        <h2 style={{ fontSize: '18px', fontWeight: 700, letterSpacing: '-0.02em', margin: 0 }}>Stateless Context Reconstruction</h2>
                    </div>
                    <ReconstructionObserver
                        sequence={requestData?.reconstruction_log?.sequence}
                        snapshot={requestData?.reconstruction_snapshot}
                    />
                </div>

                {/* Thread History */}
                {relatedRequests.length > 0 && (
                    <div style={{ marginTop: '64px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px' }}>
                            <Server size={18} className="text-secondary" />
                            <h2 style={{ fontSize: '18px', fontWeight: 700, letterSpacing: '-0.02em', margin: 0 }}>Related Thread History</h2>
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '1px', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.05)' }}>
                            {relatedRequests.map((req: any) => (
                                <div
                                    key={req.id}
                                    onClick={() => router.push(`/analytics/${req.id}`)}
                                    style={{
                                        background: '#050505',
                                        padding: '16px 24px',
                                        display: 'grid',
                                        gridTemplateColumns: '1fr 2fr 1fr 1fr',
                                        alignItems: 'center',
                                        cursor: 'pointer',
                                        transition: 'background 0.2s ease'
                                    }}
                                    className="hover-bg-muted"
                                >
                                    <div style={{ fontSize: '12px', opacity: 0.4 }}>{new Date(req.timestamp * 1000).toLocaleTimeString()}</div>
                                    <div style={{ fontSize: '13px', fontWeight: 500 }}>{req.model.split('/').pop()}</div>
                                    <div style={{ fontSize: '13px', color: 'var(--accent)' }}>{req.tokens_saved} SAVED</div>
                                    <div style={{ fontSize: '13px', textAlign: 'right', opacity: 0.5 }}>{formatMs(req.latency_ms)}</div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

            </main>
        </div>
    );
}
