"use client";

import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
    ChevronDown, ArrowRight, Clock,
    Sparkles, Layout, Search, Zap, Code
} from "lucide-react";
import { useRouter } from "next/navigation";
import { Sidebar } from "@/components/Sidebar";
import { createClient } from "@/utils/supabase";

// --- Types ---
interface AnalyticsRow {
    id: string;
    model: string;
    tokens_in: number;
    tokens_out: number;
    tokens_saved: number;
    latency_ms: number;
    timestamp: number;
    raw_messages: any[];
    response_message: any;
    reconstruction_log: {
        sequence: Array<{
            source: string;
            text: string;
            score: number;
            line_index: number;
        }>;
    };
}

export default function ObservabilityPage() {
    const router = useRouter();
    const [requests, setRequests] = useState<AnalyticsRow[]>([]);
    const [selectedRequest, setSelectedRequest] = useState<AnalyticsRow | null>(null);
    const [loading, setLoading] = useState(true);

    const GATEWAY_URL = process.env.NEXT_PUBLIC_GATEWAY_URL || "http://localhost:8000";
    const supabase = createClient();

    useEffect(() => {
        const fetchRequests = async () => {
            const { data: { user } } = await supabase.auth.getUser();
            if (user) {
                try {
                    const resp = await fetch(`${GATEWAY_URL}/api/user/stats`, {
                        headers: { "x-user-id": user.id }
                    });
                    const data = await resp.json();
                    const sorted = (data.recent_requests || []).sort((a: any, b: any) => b.timestamp - a.timestamp);
                    setRequests(sorted);
                    if (sorted.length > 0) setSelectedRequest(sorted[0]);
                } catch (e) {
                    console.error("Failed to fetch requests", e);
                }
                setLoading(false);
            }
        };
        fetchRequests();
    }, [supabase.auth]);

    return (
        <div className="layout-wrapper">
            <Sidebar />
            <main className="main-content">

                {/* Header - Matching Analytics Style */}
                <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "48px" }}>
                    <div>
                        <h1 style={{ fontSize: "24px", fontWeight: 600 }}>Deep Observability</h1>
                        <p className="text-muted" style={{ fontSize: "13px", marginTop: "4px" }}>Analyze context reconstruction and token efficiency per request.</p>
                    </div>

                    {/* Request Selection Dropdown - Styled like selection buttons */}
                    <div style={{ position: 'relative', minWidth: '320px' }}>
                        <select
                            value={selectedRequest?.id || ""}
                            onChange={(e) => setSelectedRequest(requests.find(r => r.id === e.target.value) || null)}
                            className="glass"
                            style={{
                                width: '100%',
                                appearance: 'none',
                                padding: '12px 16px',
                                borderRadius: '8px',
                                fontSize: '12px',
                                fontWeight: 600,
                                color: 'var(--fg)',
                                border: '1px solid rgba(255,255,255,0.05)',
                                cursor: 'pointer',
                                background: 'rgba(255,255,255,0.02)',
                                outline: 'none'
                            }}
                        >
                            {requests.map((r) => (
                                <option key={r.id} value={r.id}>
                                    {new Date(r.timestamp * 1000).toLocaleString()} • {r.model.split('/').pop()}
                                </option>
                            ))}
                            {requests.length === 0 && <option value="" disabled>No requests available</option>}
                        </select>
                        <ChevronDown style={{ position: 'absolute', right: '12px', top: '50%', transform: 'translateY(-50%)', opacity: 0.3, pointerEvents: 'none' }} size={14} />
                    </div>
                </header>

                <AnimatePresence mode="wait">
                    {selectedRequest ? (
                        <motion.div
                            key={selectedRequest.id}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                            transition={{ duration: 0.2 }}
                        >
                            {/* Main Bento Grid */}
                            <div style={{ display: "grid", gridTemplateColumns: "1.5fr 1fr", gap: "32px", marginBottom: "32px" }}>

                                {/* Visual Sequence Card */}
                                <div className="glass card" style={{ padding: '32px' }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '32px' }}>
                                        <div>
                                            <h3 style={{ fontSize: "12px", fontWeight: 800, color: "var(--muted)", textTransform: 'uppercase', letterSpacing: '0.05em' }}>Reconstruction Flow</h3>
                                            <div style={{ fontSize: "24px", fontWeight: 500, marginTop: "8px" }}>
                                                {selectedRequest.reconstruction_log?.sequence?.length || 0} <span style={{ opacity: 0.3, fontSize: '14px' }}>Active Chunks</span>
                                            </div>
                                        </div>
                                        <div className="savings-badge">
                                            <Zap size={12} /> {((selectedRequest.tokens_saved / (selectedRequest.tokens_in + selectedRequest.tokens_saved || 1)) * 100).toFixed(1)}% Optimized
                                        </div>
                                    </div>

                                    {/* Sequence Preview List */}
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                                        {selectedRequest.reconstruction_log?.sequence?.slice(0, 5).map((item, i) => (
                                            <div key={i} className="glass" style={{ padding: '16px', borderRadius: '12px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                                                    <div style={{
                                                        width: '32px',
                                                        height: '32px',
                                                        borderRadius: '8px',
                                                        background: item.score > 0.5 ? 'var(--accent-muted)' : 'rgba(255,255,255,0.02)',
                                                        display: 'flex',
                                                        alignItems: 'center',
                                                        justifyContent: 'center',
                                                        fontSize: '10px',
                                                        fontWeight: 800,
                                                        color: item.score > 0.5 ? 'var(--accent)' : 'rgba(255,255,255,0.2)'
                                                    }}>
                                                        {(item.score * 100).toFixed(0)}%
                                                    </div>
                                                    <span style={{ fontSize: '13px', opacity: 0.8, fontFamily: 'monospace', maxWidth: '300px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                                        {item.text}
                                                    </span>
                                                </div>
                                                <span style={{ fontSize: '10px', opacity: 0.2 }}>LINE {item.line_index}</span>
                                            </div>
                                        ))}
                                        {(!selectedRequest.reconstruction_log?.sequence || selectedRequest.reconstruction_log.sequence.length === 0) && (
                                            <div style={{ textAlign: 'center', padding: '40px', opacity: 0.3, fontSize: '13px' }}>No reconstruction log available for this request.</div>
                                        )}
                                    </div>
                                </div>

                                {/* Performance Stats Column */}
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                                    <div className="glass card stat-item" style={{ background: "var(--accent-muted)", borderColor: "var(--accent)" }}>
                                        <div className="stat-label" style={{ color: "var(--accent)" }}>Tokens Distilled</div>
                                        <div className="stat-value" style={{ color: "var(--accent)" }}>{selectedRequest.tokens_saved.toLocaleString()}</div>
                                        <div style={{ fontSize: "11px", opacity: 0.6 }}>Bypassed via reconstruction</div>
                                    </div>
                                    <div className="glass card stat-item">
                                        <div className="stat-label">Execution Latency</div>
                                        <div className="stat-value">{selectedRequest.latency_ms.toFixed(0)} <span style={{ fontSize: "14px", opacity: 0.4 }}>ms</span></div>
                                    </div>
                                    <div className="glass card stat-item">
                                        <div className="stat-label">Input Payload</div>
                                        <div className="stat-value">{selectedRequest.tokens_in.toLocaleString()} <span style={{ fontSize: "14px", opacity: 0.4 }}>tokens</span></div>
                                    </div>
                                </div>
                            </div>

                            {/* Bottom Row - Preview & Entry */}
                            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "32px" }}>
                                <div className="glass card">
                                    <h3 style={{ fontSize: "14px", fontWeight: 600, marginBottom: "24px" }}>Prompt Context</h3>
                                    <div style={{ fontSize: '13px', opacity: 0.6, background: 'rgba(255,255,255,0.02)', padding: '20px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)', minHeight: '120px' }}>
                                        {selectedRequest.raw_messages?.slice(-1)[0]?.content?.toString().slice(0, 400)}...
                                    </div>
                                </div>

                                <div className="glass card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', gap: '20px', textAlign: 'center' }}>
                                    <div style={{ width: '64px', height: '64px', borderRadius: '20px', background: 'var(--accent-muted)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                        <Layout className="text-accent" size={32} />
                                    </div>
                                    <div>
                                        <h4 style={{ fontSize: '18px', fontWeight: 600, marginBottom: '4px' }}>Analyze Complete Flow</h4>
                                        <p style={{ fontSize: '13px', opacity: 0.4 }}>Step through the full animation of how this context was built.</p>
                                    </div>
                                    <button
                                        onClick={() => router.push(`/analytics/${selectedRequest.id}`)}
                                        className="btn btn-primary"
                                        style={{ width: '100%', height: '56px', borderRadius: '16px', fontSize: '15px', fontWeight: 700 }}
                                    >
                                        Enter Observer Mode <ArrowRight size={18} />
                                    </button>
                                </div>
                            </div>

                        </motion.div>
                    ) : !loading && (
                        <div style={{ textAlign: "center", padding: "100px 0", opacity: 0.2 }}>
                            <Search size={48} style={{ margin: "0 auto 16px" }} />
                            <p>No requests found in account</p>
                        </div>
                    )}
                </AnimatePresence>

            </main>

            <style jsx global>{`
        select option {
          background: #0d0d0d !important;
          color: #fff !important;
          padding: 12px !important;
        }
      `}</style>
        </div>
    );
}
