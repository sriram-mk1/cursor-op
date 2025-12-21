"use client";

import { useState, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
import { Sidebar } from "@/components/Sidebar";
import { motion } from "framer-motion";
import {
    ArrowLeft, Shield, Database, Activity, TrendingUp, Cpu, Server, Clock, Zap
} from "lucide-react";
import { ReconstructionObserver } from "@/components/ReconstructionObserver";
import { createClient } from "@/utils/supabase";

export default function AnalyticsDetailPage() {
    const params = useParams();
    const router = useRouter();
    const [requestData, setRequestData] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const supabase = createClient();

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
                    const req = data.recent_requests?.find((r: any) => r.id === params.id);
                    if (req) setRequestData(req);
                } catch (e) {
                    console.error("Failed to fetch detail", e);
                }
            }
            setLoading(false);
        };
        fetchDetail();
    }, [params.id, GATEWAY_URL, supabase.auth]);

    // Robust Fallback Access using real DB column 'metadata'
    // 'or_data' comes from OpenRouter's generation stats endpoint
    const or_data = requestData?.metadata || {};
    const usage_cost = or_data.total_cost || requestData?.total_cost_usd || 0;

    // Formatting Helpers
    const formatCost = (val: number) => `$${(val || 0).toFixed(6)}`;
    const formatMs = (val: number) => `${Math.round(val || 0)}ms`;

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
                            <h1 style={{ fontSize: "36px", fontWeight: 700, letterSpacing: '-0.04em', lineHeight: 1, marginBottom: '8px' }}>OpenRouter Generation Report</h1>
                            <div style={{ fontFamily: 'monospace', fontSize: '13px', color: 'rgba(255,255,255,0.3)', display: 'flex', gap: '16px' }}>
                                <span>SESSION: {requestData?.session_id || 'N/A'}</span>
                                {or_data.id && <span>OR_ID: {or_data.id}</span>}
                            </div>
                        </div>
                        <div style={{ textAlign: 'right' }}>
                            <div style={{ fontSize: '11px', fontWeight: 800, letterSpacing: '0.1em', opacity: 0.4, marginBottom: '4px' }}>TOTAL COST</div>
                            <div style={{ fontSize: '24px', fontWeight: 700 }}>{formatCost(usage_cost)}</div>
                        </div>
                    </div>
                </header>

                {/* Primary Data Grid (Bento Style) - Real Real Data */}
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
                                <div style={{ fontSize: '9px', opacity: 0.4, marginBottom: '4px' }}>NATIVE CACHED</div>
                                <div style={{ fontSize: '18px', fontWeight: 700, color: 'var(--accent)' }}>{or_data.native_tokens_cached || 0}</div>
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
                                <span>Generation</span>
                                <span>{formatMs(or_data.generation_time)}</span>
                            </div>
                            {or_data.moderation_latency > 0 && (
                                <div style={{ display: 'flex', justifyContent: 'space-between', opacity: 0.6 }}>
                                    <span>Moderation</span>
                                    <span>{formatMs(or_data.moderation_latency)}</span>
                                </div>
                            )}
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
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <span style={{ opacity: 0.5 }}>Finish</span>
                                <span style={{ textTransform: 'uppercase', background: 'rgba(255,255,255,0.1)', padding: '2px 6px', borderRadius: '2px', fontSize: '9px' }}>
                                    {or_data.finish_reason || 'STOP'}
                                </span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Animation: Sequence directly from DB */}
                <ReconstructionObserver
                    sequence={requestData?.reconstruction_log?.sequence}
                    snapshot={requestData?.reconstruction_snapshot}
                />

            </main>
        </div>
    );
}
